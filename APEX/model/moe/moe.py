import dataclasses
from typing import List, Dict, Optional, Tuple
import os
import threading

import torch
import torch.nn.functional as F
from torch import nn

# 线程局部存储，用于在不同层之间传递input_features和预计算的专家索引
_thread_local = threading.local()



class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs_raw: torch.Tensor):
        ishape = inputs_raw.shape
        inputs = inputs_raw.view(-1,ishape[-1])
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        results_out = results.view(ishape)
        return results_out


def select_expert_by_anchor(input_features: Dict[str, torch.Tensor], 
                           anchors: Dict[int, Dict[str, torch.Tensor]], 
                           modality_combination: str = "all") -> (torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]):
    """
    基于锚点相似度选择专家
    
    Args:
        input_features: 包含 'image', 'text', 'audio', 'video' 特征的字典
        anchors: 包含各任务锚点的字典，格式为 {task_id: {'image': tensor, 'text': tensor, 'audio': tensor, 'video': tensor}}
        modality_combination: 模态组合方式，可选值：
            - "text_audio": ['text', 'audio']
            - "video_audio": ['video', 'audio']
            - "text_video": ['text', 'video']
            - "text": ['text']
            - "audio": ['audio']
            - "video": ['video']
            - "all": ['image', 'text', 'audio', 'video'] (默认)
    
    Returns:
        选中的专家索引、所有专家的相似度分数、每种模态的相似度详情
    """
    similarities = []
    modality_similarities = {'image': [], 'text': [], 'audio': [], 'video': []}

    if not input_features:
        num_experts = len(anchors)
        empty_modality_sims = {k: torch.zeros(num_experts) for k in ['image', 'text', 'audio', 'video']}
        return torch.tensor(0), torch.zeros(num_experts), empty_modality_sims

    device = next(iter(input_features.values())).device
    
    # 根据modality_combination确定要使用的模态
    modality_combinations = {
        "text_audio": ['text', 'audio'],
        "video_audio": ['video', 'audio'],
        "text_video": ['text', 'video'],
        "text": ['text'],
        "audio": ['audio'],
        "video": ['video'],
        "all": ['image', 'text', 'audio', 'video']
    }
    
    selected_modalities = modality_combinations.get(modality_combination, ['image', 'text', 'audio', 'video'])
    
    for task_id, task_anchors in anchors.items():
        sim_scores = []
        task_modality_sims = {'image': 0.0, 'text': 0.0, 'audio': 0.0, 'video': 0.0}
        
        for key in selected_modalities:
            input_feat = input_features.get(key)
            anchor_feat_cpu = task_anchors.get(key)

            if input_feat is not None and anchor_feat_cpu is not None:
                anchor_feat = anchor_feat_cpu.to(device)
                
                # Vectorize by taking the mean over the first dimension (patches/tokens)
                input_vec = input_feat.mean(dim=0) if input_feat.ndim > 1 else input_feat
                anchor_vec = anchor_feat.mean(dim=0) if anchor_feat.ndim > 1 else anchor_feat

                if input_vec.ndim == 0 or anchor_vec.ndim == 0:
                    task_modality_sims[key] = 0.0
                    continue

                # Ensure vectors are 1D
                input_vec = input_vec.flatten()
                anchor_vec = anchor_vec.flatten()

                # Pad to same length if necessary
                if input_vec.shape != anchor_vec.shape:
                    min_len = min(input_vec.numel(), anchor_vec.numel())
                    input_vec = input_vec[:min_len]
                    anchor_vec = anchor_vec[:min_len]

                # Use eps for numerical stability
                sim_score = F.cosine_similarity(input_vec, anchor_vec, dim=0, eps=1e-8)
                sim_scores.append(sim_score)
                task_modality_sims[key] = sim_score.item()
            else:
                task_modality_sims[key] = 0.0
        
        # 记录每种模态的相似度
        for key in ['image', 'text', 'audio', 'video']:
            modality_similarities[key].append(task_modality_sims[key])

        if sim_scores:
            similarities.append(torch.stack(sim_scores).mean())
        else:
            similarities.append(torch.tensor(0.0, device=device))
    
    if not similarities:
        num_experts = len(anchors)
        empty_modality_sims = {k: torch.zeros(num_experts) for k in ['image', 'text', 'audio', 'video']}
        return torch.tensor(0, device=device), torch.zeros(num_experts, device=device), empty_modality_sims

    similarities_tensor = torch.stack(similarities)
    selected_expert = torch.argmax(similarities_tensor)
    
    # 转换模态相似度为tensor
    modality_sims_tensor = {}
    for key in ['image', 'text', 'audio', 'video']:
        modality_sims_tensor[key] = torch.tensor(modality_similarities[key], device=device)
    
    return selected_expert, similarities_tensor, modality_sims_tensor


def precompute_expert_selection(input_features: Dict[str, torch.Tensor], 
                               anchors: Dict[int, Dict[str, torch.Tensor]], 
                               modality_combination: str = "all") -> Optional[Tuple[int, torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    预先计算专家选择，避免在每个AnchorMoeLayer中重复计算
    
    Args:
        input_features: 包含 'image', 'text', 'audio', 'video' 特征的字典
        anchors: 包含各任务锚点的字典，格式为 {task_id: {'image': tensor, 'text': tensor, 'audio': tensor, 'video': tensor}}
        modality_combination: 模态组合方式，与select_expert_by_anchor函数相同
    
    Returns:
        选中的专家索引、相似度分数和每种模态的相似度详情，如果无法选择则返回None
    """
    if not input_features or not anchors:
        return None, None, None
    
    try:
        selected_expert, similarities, modality_similarities = select_expert_by_anchor(input_features, anchors, modality_combination)
        return selected_expert, similarities, modality_similarities
    except Exception as e:
        print(f"预计算专家选择时出错: {e}")
        return None, None, None


class AnchorMoeLayer(nn.Module):
    """
    基于锚点的MoE层，用于推理阶段
    """
    def __init__(self, experts: List[nn.Module], anchors: Dict[int, Dict[str, torch.Tensor]], 
                 num_experts: int, num_experts_per_tok: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.anchors = anchors
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs_raw: torch.Tensor, input_features: Optional[Dict[str, torch.Tensor]] = None):
        ishape = inputs_raw.shape
        inputs = inputs_raw.view(-1, ishape[-1])
        
        # 优先使用预计算的专家索引
        precomputed_expert_idx = getattr(_thread_local, 'selected_expert_idx', None)
        
        if precomputed_expert_idx is not None:
            # 使用预计算的专家索引
            # 如果是tensor，需要转换为int
            if isinstance(precomputed_expert_idx, torch.Tensor):
                expert_idx = precomputed_expert_idx.item()
            else:
                expert_idx = precomputed_expert_idx
            
            # 确保索引在有效范围内
            expert_idx = max(0, min(expert_idx, len(self.experts) - 1))
            expert = self.experts[expert_idx]
            results = expert(inputs)
        else:
            print("DEBUG: 没有直接传递预计算的专家索引，回退到第一个专家")
            results = self.experts[0](inputs)
        
        results_out = results.view(ishape)
        return results_out


def load_anchors(anchor_save_path: str, num_experts: int) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    加载任务锚点
    
    Args:
        anchor_save_path: 锚点保存路径
        num_experts: 专家数量
    
    Returns:
        锚点字典
    """
    anchors = {}
    for task_id in range(num_experts):
        anchor_file = os.path.join(anchor_save_path, f"anchor_task_{task_id}.pt")
        if os.path.exists(anchor_file):
            anchors[task_id] = torch.load(anchor_file, map_location='cpu')
    return anchors


def save_anchor(anchor_data: Dict[str, torch.Tensor], anchor_save_path: str, task_id: int):
    """
    保存任务锚点
    
    Args:
        anchor_data: 锚点数据，包含 'image', 'text', 'audio' 特征
        anchor_save_path: 保存路径
        task_id: 任务ID
    """
    os.makedirs(anchor_save_path, exist_ok=True)
    anchor_file = os.path.join(anchor_save_path, f"anchor_task_{task_id}.pt")
    torch.save(anchor_data, anchor_file)


def update_anchor_incrementally(current_anchor: Dict[str, torch.Tensor], 
                              new_sample: Dict[str, torch.Tensor], 
                              sample_count: int) -> Dict[str, torch.Tensor]:
    """
    增量更新锚点
    
    Args:
        current_anchor: 当前锚点
        new_sample: 新样本特征
        sample_count: 当前样本数量
    
    Returns:
        更新后的锚点
    """
    updated_anchor = {}
    for key in current_anchor.keys():
        if key in new_sample:
            # 增量更新公式: new_mean = (n * old_mean + new_sample) / (n + 1)
            updated_anchor[key] = (sample_count * current_anchor[key] + new_sample[key]) / (sample_count + 1)
        else:
            updated_anchor[key] = current_anchor[key]
    return updated_anchor


class AnchorTracker:
    """锚点跟踪器，用于在训练过程中收集和更新锚点"""
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.anchor_data = {'image': None, 'text': None, 'audio': None, 'video': None}
        self.sample_count = 0
    
    def update_anchor(self, features: Dict[str, torch.Tensor]):
        """更新锚点数据"""
        for key, value in features.items():
            if key in self.anchor_data:
                if self.anchor_data[key] is None:
                    self.anchor_data[key] = value.detach().cpu().clone()
                else:
                    # 增量更新
                    self.anchor_data[key] = update_anchor_incrementally(
                        {key: self.anchor_data[key]}, 
                        {key: value.detach().cpu()}, 
                        self.sample_count
                    )[key]
        self.sample_count += 1
    
    def get_anchor(self):
        """获取当前锚点"""
        return {k: v for k, v in self.anchor_data.items() if v is not None}

