"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Dict, List

from ...core import register

__all__ = ['MultiModalMDAFN', ]

@register()
class MultiModalMDAFN(nn.Module):
    """
    多模态MDAFN模型：处理可见光和红外两种模态输入
    """
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, 
                 backbone: nn.Module, 
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 fusion_method: str = 'add'):
        """
        初始化多模态MDAFN模型
        
        Args:
            backbone: 骨干网络
            encoder: 编码器
            decoder: 解码器
            fusion_method: 特征融合方法，目前只支持'add'
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.fusion_method = fusion_method
        
        # 确保fusion_method有效
        assert fusion_method in ['add'], f"目前只支持'add'融合方式，不支持{fusion_method}"
        
        # EMD权重生成器：固定通道数量，避免LazyLinear引起EMA初始化问题
        # 这里假设encoder最后一层输出通道为256，如果不是请修改此值
        self._emd_feat_dim = 256  # 假设值，需要根据实际特征通道数调整
        self.emd_weight_generator = nn.Linear(self._emd_feat_dim, 2)
        # 偏置初始化为0.5，使得模型开始时两个损失权重相等
        torch.nn.init.constant_(self.emd_weight_generator.bias, 0.5)
        # 初始权重为小随机值
        torch.nn.init.normal_(self.emd_weight_generator.weight, mean=0.0, std=0.01)
        
    def forward(self, x, targets=None):
        """
        前向传播
        
        Args:
            x: 字典类型，包含可见光和红外两种模态
               格式为 {'vis': vis_images, 'ir': ir_images}
            targets: 目标标注
            
        Returns:
            模型输出
        """
        if x.shape[1] == 6:
            # 分别提取两种模态的特征
            vis_features = self.backbone(x[:, 0:3, :])
            ir_features = self.backbone(x[:, 3:6, :])

            # 构建注意力图：对最后一层特征进行通道平均并经过sigmoid映射
            vis_attn = torch.sigmoid(vis_features[-1].mean(dim=1))
            ir_attn = torch.sigmoid(ir_features[-1].mean(dim=1))

            # 后续处理
            vis_feats = self.encoder(vis_features)
            ir_feats = self.encoder(ir_features)
           
            out = self.decoder([vis_feats, ir_feats], targets)
            # 将注意力图加入输出
            out['vis_attention'] = vis_attn
            out['ir_attention'] = ir_attn
            
            # 记录特征维度以便检查是否与初始化时一致
            if self.training and hasattr(self, '_emd_feat_dim'):
                actual_feat_dim = ir_feats[-1].shape[1]
                if self._emd_feat_dim != actual_feat_dim:
                    print(f"警告: EMD权重生成器初始化维度({self._emd_feat_dim})与实际特征维度({actual_feat_dim})不一致")
            
            # 仅在训练阶段生成EMD动态权重
            if self.training:
                pooled_feat = F.adaptive_avg_pool2d(vis_feats[-1], (1, 1)).squeeze(-1).squeeze(-1)
                weights = F.softmax(self.emd_weight_generator(pooled_feat), dim=1)
                out['emd_weights'] = weights
            
            return out
        else:
            # 如果只有单个模态输入（用于兼容性）
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x, targets)
            
            return x
    
    def deploy(self):
        """部署模式转换"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 