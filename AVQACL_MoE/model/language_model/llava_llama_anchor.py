#    Modified from Copyright 2023 Haotian Liu
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

from typing import List, Optional, Tuple, Union, Dict
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig
from AVQACL_MoE.model.moe.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaMLP
from AVQACL_MoE.model.moe.moe import AnchorMoeLayer, load_anchors, _thread_local, precompute_expert_selection

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import threading

class LlavaLlamaAnchorConfig(LlamaConfig):
    model_type = "llava_llama_anchor"


class LlavaLlamaAnchorModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaAnchorConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaAnchorModel, self).__init__(config)
        self.anchor_mode = getattr(config, 'anchor_mode', False)
        self.anchors = {}
        self.num_experts = getattr(config, 'num_experts', 4)
        
    def load_expert_weights_and_anchors(self, expert_weights_path: str, anchor_save_path: str, num_experts: int):
        """加载专家权重和锚点数据"""
        # 加载锚点
        self.anchors = load_anchors(anchor_save_path, num_experts)
        
        # 确保锚点数据在正确的设备上
        device = next(self.parameters()).device
        for task_id in self.anchors:
            for modality in self.anchors[task_id]:
                self.anchors[task_id][modality] = self.anchors[task_id][modality].to(device)
        
        # 加载专家权重并替换MoE层
        expert_weights_list = []
        for task_id in range(num_experts):
            expert_file = os.path.join(expert_weights_path, f"expert_task_{task_id}.pt")
            if os.path.exists(expert_file):
                expert_weights = torch.load(expert_file, map_location='cpu')
                expert_weights_list.append(expert_weights)
            else:
                print(f"警告: 专家权重文件不存在: {expert_file}")
                expert_weights_list.append(None)
        
        # 替换指定层的MLP为AnchorMoeLayer，由于仅用于推理阶段，直接判断偶数层进行MoE替换
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx % 2 == 0:
                # 创建专家列表
                experts = []
                for task_id in range(num_experts):
                    expert = LlamaMLP(self.config)
                    if expert_weights_list[task_id] is not None:
                        # 加载对应的专家权重
                        expert_state_dict = {}
                        layer_prefix = f"model.layers.{layer_idx}.mlp."
                        for key, value in expert_weights_list[task_id].items():
                            if key.startswith(layer_prefix):
                                new_key = key[len(layer_prefix):]
                                expert_state_dict[new_key] = value
                        
                        if expert_state_dict:
                            expert.load_state_dict(expert_state_dict, strict=False)
                            print(f"任务 {task_id} 层 {layer_idx} 加载了 {len(expert_state_dict)} 个参数")
                        else:
                            print(f"警告: 任务 {task_id} 层 {layer_idx} 没有找到匹配的专家权重参数")
                            # 打印可用的参数名以便调试
                            available_keys = [k for k in expert_weights_list[task_id].keys() if 'mlp' in k][:3]
                            if available_keys:
                                print(f"  可用的MLP参数示例: {available_keys}")
                    experts.append(expert)
                
                # 替换为AnchorMoeLayer
                layer.mlp = AnchorMoeLayer(
                    experts=experts,
                    anchors=self.anchors,
                    num_experts=num_experts,
                    num_experts_per_tok=1
                )
                print(f"层 {layer_idx} 已替换为AnchorMoeLayer")
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                past_key_values=None, inputs_embeds=None, use_cache=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None,
                input_features=None, **kwargs):
        """前向传播"""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LlavaLlamaAnchorForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaAnchorConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaAnchorModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Custom generate method to handle expert selection for MoE.

        It intercepts 'selected_expert_idx' and stores it in a thread-local
        variable to be used by the AnchorMoeLayer during the forward pass.
        """
        selected_expert_idx = kwargs.pop('selected_expert_idx', None)

        # Store the selected expert index in thread-local storage
        # This makes it accessible to the AnchorMoeLayer
        _thread_local.selected_expert_idx = selected_expert_idx

        try:
            # Call the original generate method
            return super().generate(*args, **kwargs)
        finally:
            # Clean up the thread-local storage to avoid side effects
            if hasattr(_thread_local, 'selected_expert_idx'):
                del _thread_local.selected_expert_idx

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_mask: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        video_mask: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        features_mask: Optional[torch.FloatTensor] = None,
        padding_masks: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_audio_and_vision(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_mask,
                videos,
                video_mask,
                input_features,
                features_mask,
                padding_masks
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_features=input_features,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 注意：generate方法已在第135行定义，此处删除重复定义

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        # Pop multimodal inputs from kwargs
        images = kwargs.pop("images", None)
        image_mask = kwargs.pop("image_mask", None)
        videos = kwargs.pop("videos", None)
        video_mask = kwargs.pop("video_mask", None)
        input_features = kwargs.pop("input_features", None)
        padding_masks = kwargs.pop("padding_masks", None)
        features_mask = kwargs.pop("features_mask", None)

        if past_key_values is not None:
            # In subsequent generation steps, only the last token is needed
            input_ids = input_ids[:, -1:]

            # Set multimodal inputs to None to avoid reprocessing
            images = videos = input_features = padding_masks = features_mask = image_mask = video_mask = None

            # Correctly handle attention_mask for subsequent steps
            past_length = past_key_values[0][0].shape[2]
            query_len = input_ids.shape[1]
            expected_len = past_length + query_len
            if attention_mask is None or attention_mask.shape[1] != expected_len:
                attention_mask = torch.ones((input_ids.shape[0], expected_len), dtype=torch.bool, device=input_ids.device)

        elif inputs_embeds is not None and attention_mask is None:
            # First step with embeddings, create a matching attention mask if none is provided
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": images,
            "image_mask": image_mask,
            "videos": videos,
            "video_mask": video_mask,
            "input_features": input_features,
            "padding_masks": padding_masks,
            "features_mask": features_mask,
        })

        return model_inputs

AutoConfig.register("llava_llama_anchor", LlavaLlamaAnchorConfig)
AutoModelForCausalLM.register(LlavaLlamaAnchorConfig, LlavaLlamaAnchorForCausalLM)