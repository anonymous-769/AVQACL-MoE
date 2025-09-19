# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Union

import torch
import torch.nn as nn

import transformers

from AVQACL_MoE.train.llava_trainer import LLaVATrainer
from AVQACL_MoE.train.data import ModelArguments,DataArguments,TrainingArguments,make_supervised_data_module
from AVQACL_MoE import conversation as conversation_lib
from AVQACL_MoE.model import *
from AVQACL_MoE.model.moe.moe import save_anchor, update_anchor_incrementally, AnchorTracker
from AVQACL_MoE.model.moe.modeling_llama import LlamaMLP

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_llm_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'audio_tower', 'mm_audio_aligner']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_all_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk. 重写了train.py的该函数,因为不微调其他地方因此不保存"""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

# 创建自定义训练器，支持锚点收集
class AnchorLLaVATrainer(LLaVATrainer):
    def __init__(self, *args, **kwargs):
        self.anchor_tracker = kwargs.pop('anchor_tracker', None)
        self.training_args_ref = kwargs.pop('training_args_ref', None)
        super().__init__(*args, **kwargs)
    
    def training_step(self, model, inputs):
        # 在锚点模式下收集特征
        if (self.anchor_tracker is not None and 
            self.training_args_ref is not None and 
            self.training_args_ref.moe_training_mode == 'anchor'):
            
            # 提取多模态特征用于锚点计算
            with torch.no_grad():
                features = {}
                try:
                    actual_model = model.module if hasattr(model, 'module') else model
                    device = next(actual_model.parameters()).device
                    dtype = next(actual_model.parameters()).dtype

                    if 'images' in inputs and inputs['images'] is not None:
                        img_tensor = inputs['images'].to(device, dtype=dtype)
                        # 逐样本检查以过滤占位符（全零张量）
                        non_placeholder_mask = torch.any(img_tensor.view(img_tensor.shape[0], -1) != 0, dim=1)
                        if torch.any(non_placeholder_mask):
                            valid_images = img_tensor[non_placeholder_mask]
                            if valid_images.ndim == 5:
                                b, t, c, h, w = valid_images.shape
                                valid_images = valid_images.view(b * t, c, h, w)
                            image_features = actual_model.get_vision_tower()(valid_images)
                            features['image'] = image_features.mean(dim=0).detach().cpu()
                            # 释放GPU内存
                            del image_features, valid_images
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    if 'videos' in inputs and inputs['videos'] is not None:
                        video_tensor = inputs['videos'].to(device, dtype=dtype)
                        # 逐样本检查以过滤占位符（全零张量）
                        non_placeholder_mask = torch.any(video_tensor.view(video_tensor.shape[0], -1) != 0, dim=1)
                        if torch.any(non_placeholder_mask):
                            valid_videos = video_tensor[non_placeholder_mask]
                            if valid_videos.ndim == 5:
                                b, t, c, h, w = valid_videos.shape
                                valid_videos = valid_videos.view(b * t, c, h, w)
                            video_features = actual_model.get_vision_tower()(valid_videos)
                            features['video'] = video_features.mean(dim=0).detach().cpu()
                            # 释放GPU内存
                            del video_features, valid_videos
                            torch.cuda.empty_cache()

                    if 'input_features' in inputs and inputs['input_features'] is not None:
                        audio_tensor = inputs['input_features'].to(device, dtype=dtype)
                        # 逐样本检查以过滤占位符（全一张量）
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
                            features['audio'] = audio_features.mean(dim=0).detach().cpu()
                            # 释放GPU内存
                            del audio_features, valid_audio
                            if padding_mask is not None:
                                del padding_mask
                            torch.cuda.empty_cache()
                    
                    if 'input_ids' in inputs and inputs['input_ids'] is not None:
                        input_ids = inputs['input_ids']
                        vocab_size = actual_model.get_input_embeddings().num_embeddings
                        valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
                        if valid_mask.any():
                            valid_input_ids = input_ids[valid_mask]
                            if valid_input_ids.numel() > 0:
                                text_embeds = actual_model.get_input_embeddings()(valid_input_ids)
                                features['text'] = text_embeds.mean(dim=0).detach().cpu()
                                # 释放GPU内存
                                del text_embeds, valid_input_ids
                                torch.cuda.empty_cache()
                except Exception as e:
                    rank0_print(f"特征提取错误: {e}")
                    pass
            
            # 更新锚点
            if features:
                # 将特征移至CPU并分离，以释放显存
                cpu_features = {k: v.detach().cpu() for k, v in features.items()}
                self.anchor_tracker.update_anchor(cpu_features)
        
        return super().training_step(model, inputs)
        
    

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None or model_args.audio_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    # 锚点训练模式下不需要初始化MoE层，只需要标准LlamaMLP，只有在推理阶段才需要替换为Anchor MoE层
    # 不设置config.moe，这样可以避免在LlamaDecoderLayer中初始化router MoE层
    if training_args.moe_training_mode == 'anchor':
        model.config.moe = None  # 确保不初始化router MoE层
        rank0_print(f"锚点训练模式：使用标准LlamaMLP，当前任务ID: {training_args.cur_task}")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 音频塔初始化
    if hasattr(model_args, 'audio_tower') and model_args.audio_tower is not None:
        model.get_model().initialize_audio_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        audio_tower = model.get_audio_tower()
        # 强制使用 float32，因为 BEATs 音频编码器的 weight normalization 不支持 bfloat16
        audio_tower.to(dtype=torch.float32, device=training_args.device)
        data_args.audio_processor = audio_tower.audio_processor
        data_args.language = model_args.language
        data_args.is_multimodal = True
        model.config.tune_mm_audio_aligner = training_args.tune_mm_audio_aligner = model_args.tune_mm_audio_aligner
        model.config.tune_mm_audio_projector = training_args.tune_mm_audio_projector = model_args.tune_mm_audio_projector
        if model_args.tune_mm_audio_projector:
            model.requires_grad_(False)
            for p in model.get_model().mm_audio_aligner.projector.parameters():
                p.requires_grad = True
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    # 锚点模式特殊处理
    expert_modules = {}  # 存储专家层模块的映射
    expert_param_names = {}  # 存储专家层参数名称映射 {layer_num: {param_name: full_param_name}}
    if training_args.moe_training_mode == 'anchor':
        rank0_print(f"启用锚点训练模式，当前任务ID: {training_args.cur_task}")
        rank0_print("锚点训练模式：使用标准LlamaMLP层，layer_num % 2 == 0的层将作为专家层保存")
        # 锚点训练模式下，模型已经是标准的LlamaMLP结构
        # 不需要进行MoE层替换，直接标记哪些层是专家层即可
        expert_layer_count = 0
        marked_layers = set()  # 用于避免重复标记
        
        # 一次性收集所有专家层信息，包括模块引用和参数名称映射
        for name, module in model.named_modules():
            if 'layers.' in name and '.mlp' in name and name.count('.') >= 3:
                # 从模块名称中提取层号，格式通常为 model.layers.X.mlp 或 base_model.model.layers.X.mlp
                parts = name.split('.')
                layers_idx = -1
                for i, part in enumerate(parts):
                    if part == 'layers':
                        layers_idx = i
                        break
                if layers_idx != -1 and layers_idx + 1 < len(parts):
                    try:
                        layer_num = int(parts[layers_idx + 1])
                        if layer_num % 2 == 0 and layer_num not in marked_layers:
                            # 标记为专家层（用于后续保存）
                            marked_layers.add(layer_num)
                            expert_layer_count += 1
                            expert_modules[layer_num] = module  # 保存模块引用
                            
                            # 同时收集该层的所有参数名称映射
                            expert_param_names[layer_num] = {}
                            for param_name, param in module.named_parameters():
                                full_param_name = f"{name}.{param_name}"
                                expert_param_names[layer_num][param_name] = full_param_name
                            
                            rank0_print(f"层 {layer_num} 标记为专家层 (任务 {training_args.cur_task})")
                    except (ValueError, IndexError):
                        continue
        
        rank0_print(f"总共标记了 {expert_layer_count} 个专家层")
        # 初始化锚点跟踪器
        anchor_tracker = AnchorTracker(training_args.cur_task)

    if training_args.llm_lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_llm_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # 获取实际的模型对象，处理PEFT包装的情况
    def get_actual_model(model):
        if hasattr(model, 'base_model'):
            # PEFT模型，需要通过base_model访问
            return model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
        else:
            # 普通模型，直接调用get_model()
            return model.get_model()
    
    actual_model = get_actual_model(model)
    
    # 参数设置现在在模块初始化时正确处理，这里不再需要重复设置
    # if model_args.tune_mm_mlp_adapter:
    #     for p in actual_model.mm_projector.parameters():
    #         p.requires_grad = True
    # if model_args.tune_mm_audio_projector:
    #     for p in actual_model.mm_audio_aligner.projector.parameters():
    #         p.requires_grad = True
    
    # Anchor moe 模型训练阶段去掉了原始的 router moe
    # for pname,p in actual_model.named_parameters():
    #     if "mlp.gate." in pname:
    #         p.requires_grad = True
    
    training_args.lora_enable = training_args.llm_lora_enable

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # 旧的集中冻结逻辑已删除，改回与 train.py 一致的按需冻结策略
    trainable_params_count = 0
    

    audio_data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                              data_args=data_args)
    model.cuda()
    
    # 确保audio_tower在model.cuda()后仍保持float32精度
    if hasattr(model_args, 'audio_tower') and model_args.audio_tower is not None:
        audio_tower = model.get_audio_tower()
        if audio_tower is not None:
            audio_tower.to(dtype=torch.float32, device=training_args.device)
    
    # Unfreeze expert layers for anchor training (moved here to ensure parameters are on the correct device)
    if training_args.moe_training_mode == 'anchor' and expert_modules:
        expert_param_count = 0
        for layer_num, module in expert_modules.items():
            module.requires_grad_(True)
            expert_param_count += sum(p.numel() for p in module.parameters())
        trainable_params_count += expert_param_count
        rank0_print(f"Unfroze {len(expert_modules)} expert layers (post device move), total {expert_param_count} parameters.")
        # 额外确保视觉/音频 projector 在需要时可训练
        if model_args.tune_mm_mlp_adapter and hasattr(actual_model, "mm_projector"):
            for p in actual_model.mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_audio_projector and hasattr(actual_model, "mm_audio_aligner"):
            for p in actual_model.mm_audio_aligner.projector.parameters():
                p.requires_grad = True

    mismatched_params = []
    target_device = torch.device(training_args.device)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.device != target_device:
                mismatched_params.append(n)
                p.data = p.data.to(target_device)
    if mismatched_params:
        rank0_print(f"已将 {len(mismatched_params)} 个可训练参数移动到 {target_device}：", mismatched_params[:10], "...")
    
    # 如日后重新启用 foreach CUDA kernel 时，可能需要统一 dtype。
    # 目前优化器已设定 foreach=False，故跳过强制转换以节省显存。
    pass
    # model.to(torch.float32) # 需要保证与输出文件的dtype一致
    
    trainer_kwargs = {
        'model': model,
        'tokenizer': tokenizer,
        'args': training_args,
        **audio_data_module
    }
    
    if training_args.moe_training_mode == 'anchor':
        trainer_kwargs['anchor_tracker'] = anchor_tracker
        trainer_kwargs['training_args_ref'] = training_args
        trainer = AnchorLLaVATrainer(**trainer_kwargs)
    else:
        trainer = LLaVATrainer(**trainer_kwargs)
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"\n{'='*40}")
    rank0_print(f"模型总参数量: {total_params / 1e6:.2f}M")
    rank0_print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    rank0_print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")
    rank0_print(f"\n--- 可训练参数列表 ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
    rank0_print(f"{'='*40}\n")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # 保存专家权重和锚点
    if training_args.moe_training_mode == 'anchor' and training_args.cur_task is not None:
        # 保存锚点
        if training_args.anchor_save_path:
            anchor_data = anchor_tracker.get_anchor()
            if anchor_data:
                save_anchor(anchor_data, training_args.anchor_save_path, training_args.cur_task)
                rank0_print(f"锚点已保存到: {training_args.anchor_save_path}/anchor_task_{training_args.cur_task}.pt")
        
        # 保存专家权重（复用之前收集的参数名称映射）
        if training_args.expert_weights_save_path:
            os.makedirs(training_args.expert_weights_save_path, exist_ok=True)
            expert_weights = {}
            model_state_dict = model.state_dict()
            
            for layer_num, param_mapping in expert_param_names.items():
                for param_name, full_param_name in param_mapping.items():
                    if full_param_name in model_state_dict:
                        # 构建标准化的参数名称，格式为 model.layers.X.mlp.xxx
                        standard_name = f"model.layers.{layer_num}.mlp.{param_name}"
                        expert_weights[standard_name] = model_state_dict[full_param_name].detach().cpu().clone()

            expert_file = os.path.join(training_args.expert_weights_save_path, f"expert_task_{training_args.cur_task}.pt")
            torch.save(expert_weights, expert_file)
            rank0_print(f"专家权重已保存到: {expert_file}，共保存 {len(expert_weights)} 个参数")
            
            # 打印保存的参数名称以便调试
            if len(expert_weights) > 0:
                rank0_print("保存的专家权重参数名称:")
                for i, key in enumerate(list(expert_weights.keys())[:10]):  # 只打印前10个作为示例
                    rank0_print(f"  {key}")
                if len(expert_weights) > 10:
                    rank0_print(f"  ... 还有 {len(expert_weights) - 10} 个参数")

    if training_args.llm_lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        safe_save_all_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    else:
        safe_save_all_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()