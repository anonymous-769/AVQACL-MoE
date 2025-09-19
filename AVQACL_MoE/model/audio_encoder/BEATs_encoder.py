import torch
import torch.nn as nn

from .BEATs import BEATs, BEATsConfig, BEATsProcessor

class BEATsAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.audio_split_type_dim = 3
        # self.language=args.language
        # self.task=args.task
        # print(args.task,args.language)
        self.local_files_only=args.local_files_only

        if not delay_load:
            self.load_model()
        else:
            checkpoint = torch.load(self.audio_tower_name)
            self.cfg_only = BEATsConfig(checkpoint['cfg'])

    def load_model(self):
        print("load audio tower from:",self.audio_tower_name)
        # print(self.task,self.language)
        # print("audio_tower:loading model",self.audio_tower_name)
        self.audio_processor = BEATsProcessor()
        checkpoint = torch.load(self.audio_tower_name)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.config = cfg
        
        # 强制使用float32创建模型以支持weight normalization
        # 临时设置默认数据类型为float32以确保weight_norm兼容性
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self.audio_tower = BEATs(cfg)
            self.audio_tower.load_state_dict(checkpoint['model'])
            # 确保模型在float32精度下
            self.audio_tower = self.audio_tower.to(dtype=torch.float32)
        finally:
            # 恢复原始默认数据类型
            torch.set_default_dtype(original_dtype)
        
        # debug
        # for k,v in self.audio_tower.named_parameters():
        #     if "conv2.weight" in k:
        #         print("aaaa",k,v)
        self.audio_tower.requires_grad_(False)
        # for name, parameter in self.audio_tower.named_parameters():
        #     print(name)
        #     print(parameter)
        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs[self.select_layer]
        # if self.select_feature == 'patch':
        #     audio_features = audio_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     audio_features = audio_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, audio, padding_mask):
        if audio.dim() == self.audio_split_type_dim:
            # 确保audio_tower始终保持float32精度以支持weight_norm
            # 检查模型参数的数据类型而不是模型本身的dtype属性
            first_param = next(self.audio_tower.parameters())
            if first_param.dtype != torch.float32:
                self.audio_tower = self.audio_tower.to(dtype=torch.float32)
            
            # 确保输入数据类型与模型权重类型一致
            original_audio_dtype = audio.dtype
            if audio.dtype != torch.float32:
                audio = audio.to(dtype=torch.float32)
            if padding_mask is not None and padding_mask.dtype != torch.float32:
                padding_mask = padding_mask.to(dtype=torch.float32)
            
            audio_forward_out = self.audio_tower.extract_features(audio, padding_mask=padding_mask)
            audio_features = self.feature_select(audio_forward_out).to(original_audio_dtype)
        else:
            raise ValueError("Fbank feature wrong dimension.")
        return audio_features

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.audio_tower.dtype

    # @property
    # def device(self):
    #     return self.audio_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.audio_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.encoder_embed_dim

    # @property
    # def num_patches(self):
    #     return (self.config.audio_size // self.config.patch_size) ** 2
