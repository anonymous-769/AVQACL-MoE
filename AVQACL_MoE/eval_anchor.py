# Modified from Uni-MoE
import os 
import sys 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Union
import io
import random
import argparse

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import torch

import transformers

from torch.utils.data import Dataset

from AVQACL_MoE import conversation as conversation_lib
from AVQACL_MoE.model import *
from AVQACL_MoE.mm_utils import tokenizer_image_audio_video_token
from AVQACL_MoE.model.all_builder import load_all_pretrained_model
from AVQACL_MoE.constants import IGNORE_INDEX
import zipfile
from PIL import Image
import copy

# Import shared classes and functions from train/data.py
from AVQACL_MoE.eval import make_supervised_data_module, args, preprocess, rank0_print, extract_image_from_zip
from AVQACL_MoE.train.data import ModelArguments as BaseModelArguments

AUDIOSTART = "/path/to/"

local_rank = None


@dataclass
class ModelArguments(BaseModelArguments):
    num_experts: Optional[int] = field(default=4, metadata={"help": "专家数量"})
    anchor_save_path: Optional[str] = field(default=None, metadata={"help": "锚点保存路径"})
    expert_weights_save_path: Optional[str] = field(default=None, metadata={"help": "专家权重保存路径"})
    use_anchor_mode: bool = field(default=False, metadata={"help": "是否使用锚点模式"})


def extract_features_for_anchor_comparison(inputs, model, tokenizer):
    """提取用于与专家锚点对比的数据特征，与train_anchor_moe.py中的方式完全一致"""
    features = {}
    device = model.device
    
    with torch.no_grad():
        # 处理 DeepSpeed 包装的模型
        actual_model = model.module if hasattr(model, 'module') else model
        dtype = next(actual_model.parameters()).dtype
        
        if 'images' in inputs and inputs['images'] is not None:
            img_tensor = inputs['images'].to(device, dtype=dtype)
            non_placeholder_mask = torch.any(img_tensor.view(img_tensor.shape[0], -1) != 0, dim=1)
            if torch.any(non_placeholder_mask):
                valid_images = img_tensor[non_placeholder_mask]
                if valid_images.ndim == 5:
                    b, t, c, h, w = valid_images.shape
                    valid_images = valid_images.view(b * t, c, h, w)
                image_features = actual_model.get_vision_tower()(valid_images)
                features['image'] = image_features.mean(dim=0)
        
        if 'videos' in inputs and inputs['videos'] is not None:
            print(f"  Debug: inputs['videos'] shape: {inputs['videos'].shape}, dtype: {inputs['videos'].dtype}")
            print(f"  Debug: inputs['videos'] numel: {inputs['videos'].numel()}")
            video_tensor = inputs['videos'].to(device, dtype=dtype)
            non_placeholder_mask = torch.any(video_tensor.view(video_tensor.shape[0], -1) != 0, dim=1)
            if torch.any(non_placeholder_mask):
                valid_videos = video_tensor[non_placeholder_mask]
                if valid_videos.ndim == 5:
                    b, t, c, h, w = valid_videos.shape
                    valid_videos = valid_videos.view(b * t, c, h, w)
                video_features = actual_model.get_vision_tower()(valid_videos)
                features['video'] = video_features.mean(dim=0)

        if 'input_features' in inputs and inputs['input_features'] is not None:
            audio_tensor = inputs['input_features'].to(device, dtype=dtype)
            non_placeholder_mask = torch.any(audio_tensor.view(audio_tensor.shape[0], -1) != 1, dim=1)
            if torch.any(non_placeholder_mask):
                valid_audio = audio_tensor[non_placeholder_mask]
                padding_mask = inputs.get('padding_masks', None)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)[non_placeholder_mask]

                if valid_audio.ndim == 5:
                    b, n, c, f, d = valid_audio.shape
                    valid_audio = valid_audio.view(b * n, c, f, d).squeeze(1)
                    if padding_mask is not None:
                        padding_mask = padding_mask.view(b * n, -1)
                audio_features = actual_model.get_audio_tower()(valid_audio, padding_mask=padding_mask)
                features['audio'] = audio_features.mean(dim=0)

        # 文本特征提取，与训练时一致
        if 'input_ids' in inputs and inputs['input_ids'] is not None:
            input_ids = inputs['input_ids'].to(device)
            # 添加有效性检查，确保input_ids在有效范围内
            vocab_size = actual_model.get_input_embeddings().num_embeddings
            valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
            
            if valid_mask.any():
                # 只使用有效的input_ids
                valid_input_ids = torch.where(valid_mask, input_ids, torch.zeros_like(input_ids))
                text_features = actual_model.get_input_embeddings()(valid_input_ids)
                # 使用valid_mask来计算平均值，忽略无效的token
                masked_features = text_features * valid_mask.unsqueeze(-1)
                # 计算所有batch和sequence维度的平均值
                valid_count = valid_mask.sum().clamp(min=1)
                features['text'] = masked_features.sum(dim=(0, 1)) / valid_count
    return features


def eval_anchor(margs):
    print("\n" + "="*50)
    print(f"开始评估，数据路径: {margs.data_path}")
    print(f"模型路径: {margs.model_path}")
    print(f"使用锚点模式: {margs.use_anchor_mode}")
    print(f"数据类型: {margs.data_type}")
    print("="*50 + "\n")
    
    # 创建参数对象
    model_args = ModelArguments()
    data_args = args()
    
    # 从命令行参数更新模型参数
    model_args.model_name_or_path = margs.model_path
    model_args.version = margs.version
    model_args.num_experts = margs.num_experts
    model_args.anchor_save_path = margs.anchor_save_path
    model_args.expert_weights_save_path = margs.expert_weights_save_path
    model_args.use_anchor_mode = margs.use_anchor_mode
    
    # 设置vision和audio tower路径（覆盖默认值）
    model_args.vision_tower = "checkpoints/clip-vit-large-patch14-336"
    model_args.audio_tower = "checkpoints/BEATs_iter3_plus_AS2M.pt"
    # 其他参数已在ModelArguments类中设置了与train_anchor_moe.py一致的默认值
    
    data_args.data_path = margs.data_path
    
    
    # 加载tokenizer和processors，确保与eval.py一致
    print("正在加载tokenizer和processors...")
    tokenizer, base_model, image_processor, audio_processor, _ = load_all_pretrained_model(
        model_path=model_args.model_name_or_path,
        model_base=None, 
        model_name="unimoe_anchor", # modified from "unimoe_lora"
        load_8bit=False, load_4bit=False, 
        vison_tower_path=model_args.vision_tower,
        audio_tower_path=model_args.audio_tower,
    )
    print("Tokenizer, processors以及基础model加载成功")

    # 创建锚点模型实例
    print("正在创建锚点模型...")
    from AVQACL_MoE.model.language_model.llava_llama_anchor import LlavaLlamaAnchorForCausalLM
    
    # 使用基础模型的配置创建锚点模型
    config = base_model.config
    config.anchor_mode = True
    config.num_experts = model_args.num_experts
    
    # 创建锚点模型并复制基础模型的权重
    model = LlavaLlamaAnchorForCausalLM(config)
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # 初始化vision和audio模块用于专家选择的特征提取
    if hasattr(model_args, 'vision_tower') and model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            vision_tower.load_model()
            vision_tower.to(device='cuda', dtype=torch.float32)
    
    if hasattr(model_args, 'audio_tower') and model_args.audio_tower is not None:
        model.get_model().initialize_audio_modules(model_args=model_args)
        audio_tower = model.get_audio_tower()
        if audio_tower is not None and not audio_tower.is_loaded:
            audio_tower.load_model()
            audio_tower.to(device='cuda', dtype=torch.float32)
    
    # 如果启用锚点模式，加载专家权重和锚点
    if model_args.use_anchor_mode:
        print(f"正在加载专家权重和锚点...")
        print(f"专家权重路径: {model_args.expert_weights_save_path}")
        print(f"锚点路径: {model_args.anchor_save_path}")
        print(f"专家数量: {model_args.num_experts}")
        model.get_model().load_expert_weights_and_anchors(
            expert_weights_path=model_args.expert_weights_save_path,
            anchor_save_path=model_args.anchor_save_path,
            num_experts=model_args.num_experts
        )
        print("专家权重和锚点加载完成")
    model.to(dtype=torch.float32)
    model.cuda()
    model.eval()
    print("锚点模型创建和加载成功")

    # 初始化多模态组件
    data_args.image_processor = image_processor
    data_args.audio_processor = audio_processor
    
    # 创建数据集
    print(f"正在创建数据集，数据路径: {data_args.data_path}...")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    eval_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    train_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers = 2
    )
    outlist = []
    if "video" in margs.data_type :
        kwargs = dict(do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024)
    if "vqa" in margs.data_type or "mmbench" in margs.data_type:
        kwargs = dict(do_sample=False,
                    num_beams=1,
                    temperature=0,
                    max_new_tokens=512)
    elif "clothoaqa" in margs.data_type:
        kwargs = dict(do_sample=True,
                    num_beams=4,
                    temperature=0.9,
                    max_new_tokens=512)
    elif "clothov" in margs.data_type:
        kwargs = dict(do_sample=True,
                num_beams=5,
                length_penalty=1.,
                temperature=1.,
                min_new_tokens=10,
                max_new_tokens=512)
    else:
        kwargs = dict(do_sample=True,
                temperature=0.2,
                max_new_tokens=1024)


    
    for step, batch in tqdm(enumerate(train_loader)):
        print(f"\n处理批次 {step+1}/{len(train_loader)}")
        with torch.no_grad():
            # 初始化专家选择索引
            selected_expert_idx = None
            # 锚点模式下预计算专家选择
            if model_args.use_anchor_mode:
                print("提取特征用于专家选择...")
                # 提取特征用于专家选择
                try:
                    input_features = extract_features_for_anchor_comparison(batch, model, tokenizer)
                    print("特征提取成功")
                    for k, v in input_features.items():
                        print(f"  {k}_features: shape={v.shape}, dtype={v.dtype}")
                        if v.numel() > 0:
                            print(f"    值范围: min={v.min().item():.4f}, max={v.max().item():.4f}")
                except Exception as e:
                    print(f"特征提取失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                print("预计算专家选择...")
                from AVQACL_MoE.model.moe.moe import precompute_expert_selection, AnchorMoeLayer
                
                # 从模型层中获取锚点
                anchors = None
                for layer in model.model.layers:
                    if isinstance(layer.mlp, AnchorMoeLayer):
                        anchors = layer.mlp.anchors
                        break
                
                if anchors:
                    print(f"锚点数量: {len(anchors)}")
                    print(f"使用模态组合: {margs.modality_combination}")
                    selected_expert_idx, similarities, modality_similarities = precompute_expert_selection(input_features, anchors, margs.modality_combination)
                    if selected_expert_idx is not None:
                        print(f"专家选择成功: 选择专家索引={selected_expert_idx.item()}")
                        if similarities is not None:
                            print(f"平均相似度: {[f'{s.item():.4f}' for s in similarities]}")
                        if modality_similarities is not None:
                            print("各模态相似度:")
                            for modality in ['image', 'text', 'audio', 'video']:
                                if modality in modality_similarities:
                                    modality_sims = modality_similarities[modality]
                                    print(f"  {modality}: {[f'{s.item():.4f}' for s in modality_sims]}")
                    else:
                        print("专家选择失败，将使用默认专家(索引0)")
                        selected_expert_idx = torch.tensor(0)
                else:
                    print("模型没有锚点，使用默认专家(索引0)")
                    selected_expert_idx = torch.tensor(0)
            
            if selected_expert_idx is not None:
                print(f"最终选择的专家: {selected_expert_idx.item()}")

            print("开始生成...")
            # print(f"生成参数: {kwargs}")
            generate_kwargs = {
                **kwargs,
                'input_ids': batch["input_ids"].to(device=model.device),
                'attention_mask': None,
                'images': batch["images"].to(device=model.device).float() if batch["images"] is not None else None,
                'image_mask': batch['image_mask'].to(device=model.device),
                'videos': batch["videos"].to(device=model.device).float() if batch["videos"] is not None else None,
                'video_mask': batch['video_mask'].to(device=model.device) if batch['video_mask']is not None else None,
                'input_features': batch["input_features"].to(device=model.device).float(),
                'padding_masks': batch['padding_masks'].to(device=model.device),
                'features_mask': batch["features_mask"].to(device=model.device),
            }
            
            # 只在锚点模式下传递selected_expert_idx
            if model_args.use_anchor_mode and selected_expert_idx is not None:
                generate_kwargs['selected_expert_idx'] = selected_expert_idx
            
            output_ids = model.generate(**generate_kwargs)
            print("解码输出...")
            outputs = tokenizer.batch_decode(output_ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            print(f"解码成功，输出长度: {len(outputs)}")
            dic = batch["data_ori"][0]
            dic["text"] = str(outputs)
            print(f"结果: {dic}")
            outlist.append(dic)

    
    # 保存结果
    print("\n保存结果...")
    # 确保输出目录存在
    output_path = margs.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(outlist, f, indent=4)
    
    print(f"评估完成，结果已保存到 {output_path}")
    print(f"处理了 {len(outlist)}/{len(train_loader)} 个样本")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="评估数据路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件名")
    parser.add_argument("--data_type", type=str, default="general", help="数据类型")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--version", type=str, default="v1", help="模型版本")
    parser.add_argument("--num_experts", type=int, default=4, help="专家数量")
    parser.add_argument("--anchor_save_path", type=str, help="锚点保存路径")
    parser.add_argument("--expert_weights_save_path", type=str, help="专家权重保存路径")
    parser.add_argument("--use_anchor_mode", action="store_true", help="是否使用锚点模式")
    parser.add_argument("--modality_combination", type=str, default="all", 
                       choices=["text_audio", "video_audio", "text_video", "text", "audio", "video", "all"],
                       help="模态组合方式: text_audio, video_audio, text_video, text, audio, video, all")
    
    margs = parser.parse_args()
    eval_anchor(margs)