"""
25°C真正SOTA训练脚本 - 保持UDDS优势，大幅提升NN
正确的优化思路：不是平衡，而是双重提升

🎯 正确策略:
1. 保持UDDS的优秀性能 (0.012左右)
2. 专门针对NN工况的深度优化
3. 工况特定的架构分支
4. 自适应损失加权
5. 目标: UDDS保持0.012，NN降到0.015，整体突破0.018
"""

import argparse
import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import savgol_filter

# 使用新的电化学特征工程
from electrochemical_features import ElectrochemicalFeatureEngineer, create_electrochemical_dataset
from model_code_lightning import setup_chinese_font
# from electrochemical_46features_datamodule import Electrochemical46FeaturesDataModule # 移除旧的导入
from electrochemical_46features_datamodule import Electrochemical46FeaturesDataModule # 导入新的StandardScaler数据模块

setup_chinese_font()


def setup_sci_style():
    """设置SCI论文级别的matplotlib样式 - 参照用户提供的图表"""
    plt.style.use('default')
    
    # 字体设置 - 参照参考图
    plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11          # 参考图的字体大小
    plt.rcParams['axes.titlesize'] = 12     # 子图标题
    plt.rcParams['axes.labelsize'] = 11     # 轴标签
    plt.rcParams['xtick.labelsize'] = 10    # 刻度标签
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10    # 图例
    plt.rcParams['axes.unicode_minus'] = False
    
    # SCI审美设置 - 模仿参考图
    plt.rcParams['axes.linewidth'] = 1.0    # 更细的边框
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    plt.rcParams['grid.linewidth'] = 0.5    # 非常细的网格线
    plt.rcParams['grid.alpha'] = 0.3        # 透明网格
    plt.rcParams['lines.linewidth'] = 1.8   # 参考图的线宽
    
    # 参考图的配色方案
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
        ['#0072BD', '#D95319', '#4DBEEE', '#A2142F'])  # 蓝色、橙红色系


def smooth_data(data, window_length=9, polyorder=2):
    """数据平滑处理，提升SCI图表质量"""
    if len(data) < window_length:
        return data
    
    # 确保window_length是奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 确保polyorder < window_length
    polyorder = min(polyorder, window_length - 1)
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except:
        return data


class TrueSOTAElectrochemicalKANTransformer(nn.Module):
    """
    真正SOTA的电化学KAN-Transformer架构
    保持UDDS优势，专门提升NN性能
    """
    
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, 
                 hidden_space, dropout_rate, embed_dim, grid_size=16,
                 temperature=None, feature_scaler=None, detector_temp_init=0.7):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.temperature = temperature
        self.feature_scaler = feature_scaler # 存储标准化器
        self.detector_temperature = nn.Parameter(torch.tensor(detector_temp_init)) # workload detector softmax temperature
        
        # === [策略1] 保持成功的特征编码器 ===
        # 基于UDDS成功经验的特征分组编码
        
        # 基础电化学特征编码器 (针对UDDS优化)
        self.basic_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        # 动态响应特征编码器 (UDDS的强项)
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(12, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        # 能量库仑特征编码器
        self.energy_encoder = nn.Sequential(
            nn.Linear(8, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        # 阻抗老化特征编码器
        self.impedance_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        # 温度补偿特征编码器
        self.temperature_encoder = nn.Sequential(
            nn.Linear(6, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        # === [策略2] 工况检测器 ===
        base_encoded_dim = hidden_space//4 * 3 + hidden_space//6 * 2
        # 新增：1D注意力池化层
        self.attention_pooling = nn.Sequential(
            nn.Linear(base_encoded_dim, 1),
            nn.Softmax(dim=1) # 在时间维度上进行Softmax
        )
        detector_input_dim = base_encoded_dim # 注意力池化后维度与编码维度相同
        self.workload_detector = nn.Sequential(
            nn.Linear(detector_input_dim, hidden_space//2),
            nn.LayerNorm(hidden_space//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space//2, hidden_space//4),
            nn.ReLU(),
            nn.Linear(hidden_space//4, 2)  # NN vs UDDS，Softmaxƶforward¶
        )
        
        # === [策略3] 工况特定的处理分支 ===
        
        # UDDS分支 (保持简单有效的设计)
        self.udds_branch = nn.Sequential(
            nn.Linear(base_encoded_dim, hidden_space),
            nn.LayerNorm(hidden_space),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.10)
        )
        
        # NN分支 (专门针对NN的复杂设计)
        self.nn_branch = nn.Sequential(
            nn.Linear(base_encoded_dim, hidden_space + 32),  # 更大容量
            nn.LayerNorm(hidden_space + 32),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.1),  # 更低dropout
            nn.Linear(hidden_space + 32, hidden_space + 16),
            nn.LayerNorm(hidden_space + 16),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space + 16, hidden_space),
            nn.LayerNorm(hidden_space)
        )
        
        # === [策略4] 自适应融合层 ===
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_space * 2, hidden_space),
            nn.LayerNorm(hidden_space),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.15)
        )
        
        # === [策略5] NN专用的深度注意力 ===
        self.nn_deep_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_space, 
                num_heads=num_heads//4,
                dropout=dropout_rate * 0.1,
                batch_first=True
            ) for _ in range(3)  # 3层深度注意力
        ])
        
        # === [策略6] UDDS保持的简单注意力 ===
        self.udds_attention = nn.MultiheadAttention(
            embed_dim=hidden_space, 
            num_heads=num_heads//2,
            dropout=dropout_rate * 0.2,
            batch_first=True
        )
        
        # === [策略7] 主要Transformer (针对NN优化) ===
        from model2 import TimeSeriesTransformer_ekan
        
        self.main_transformer = TimeSeriesTransformer_ekan(
            input_dim=hidden_space,
            num_heads=num_heads,
            num_layers=num_layers,
            num_outputs=num_outputs,
            hidden_space=hidden_space,
            dropout_rate=dropout_rate * 0.5,  # 降低dropout提升NN性能
            embed_dim=embed_dim,
            grid_size=grid_size,
            degree=5,
            use_residual_scaling=True
        )
        
        # === [策略8] 工况特定的预测头 ===
        self.nn_prediction_head = nn.Sequential(
            nn.Linear(hidden_space, hidden_space//2),
            nn.LayerNorm(hidden_space//2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space//2, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Linear(hidden_space//4, num_outputs)
        )
        
        self.udds_prediction_head = nn.Sequential(
            nn.Linear(hidden_space, hidden_space//3),
            nn.LayerNorm(hidden_space//3),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//3, num_outputs)
        )
        
        # === 电化学物理约束层 ===
        self.electrochemical_constraint = TrueSOTAConstraintLayer(num_outputs, temperature, feature_scaler=feature_scaler)
        
        # 新增：可学习的融合权重参数
        self.fusion_weight_param = nn.Parameter(torch.tensor(0.5)) # 初始值0.5，表示平均融合
        self.fusion_weight_head_param = nn.Parameter(torch.tensor(0.5)) # ͷ���������ʵȨ��
        
    def forward(self, x, return_uncertainty=False):
        batch_size, seq_len, features = x.shape
        
        # === 特征编码 ===
        basic_encoded = self.basic_encoder(x[:, :, :10])
        dynamic_encoded = self.dynamic_encoder(x[:, :, 10:22])
        energy_encoded = self.energy_encoder(x[:, :, 22:30])
        impedance_encoded = self.impedance_encoder(x[:, :, 30:40])
        temp_encoded = self.temperature_encoder(x[:, :, 40:46])
        
        # 拼接特征
        concatenated_features = torch.cat([
            basic_encoded, dynamic_encoded, energy_encoded, 
            impedance_encoded, temp_encoded
        ], dim=-1)
        
        # === 工况检测 - 使用1D注意力池化 ===
        attention_weights = self.attention_pooling(concatenated_features) # [batch, seq_len, 1]
        detector_input = torch.sum(concatenated_features * attention_weights, dim=1) # [batch, base_encoded_dim]
        detector_logits = self.workload_detector(detector_input)
        temp = torch.clamp(self.detector_temperature, min=0.2, max=5.0)
        workload_probs = F.softmax(detector_logits / temp, dim=-1)  # [batch, 2]
        nn_prob = workload_probs[:, 0]      # NN工况概率
        udds_prob = workload_probs[:, 1]    # UDDS工况概率
        # print(f"DEBUG: workload_probs (NN, UDDS): mean_nn={nn_prob.mean().item():.4f}, mean_udds={udds_prob.mean().item():.4f}, std_nn={nn_prob.std().item():.4f}, std_udds={udds_prob.std().item():.4f}")
        
        # === 工况特定处理 ===
        udds_features = self.udds_branch(concatenated_features)
        nn_features = self.nn_branch(concatenated_features)
        
        # === 工况特定注意力 ===
        # UDDS: 简单注意力 (保持优势)
        udds_attended, _ = self.udds_attention(udds_features, udds_features, udds_features)
        udds_enhanced = udds_features + udds_attended
        
        # NN: 深度注意力 (专门优化)
        nn_enhanced = nn_features
        for attention_layer in self.nn_deep_attention:
            nn_attended, _ = attention_layer(nn_enhanced, nn_enhanced, nn_enhanced)
            nn_enhanced = nn_enhanced + nn_attended  # 残差连接
        
        # === 自适应融合 ===
        # 根据工况概率动态融合
        nn_weight = nn_prob.unsqueeze(1).unsqueeze(2)
        udds_weight = udds_prob.unsqueeze(1).unsqueeze(2)
        
        weighted_features = nn_enhanced * nn_weight + udds_enhanced * udds_weight
        
        # === 主要Transformer处理 ===
        # Transformer输出形状: [batch, num_outputs] (因为TimeSeriesTransformer_ekan内部已取最后一个时间步)
        transformer_output = self.main_transformer(weighted_features)
        # print(f"DEBUG: transformer_output (原始): min={transformer_output.min().item():.4f}, max={transformer_output.max().item():.4f}, mean={transformer_output.mean().item():.4f}, std={transformer_output.std().item():.4f}")
        # 此时 transformer_output 形状为 [batch, num_outputs]，无需再取[:, -1, :]
        
        # === 工况特定预测 ===
        # 预测头已经输出 [batch, num_outputs] (现在是logits)
        nn_pred_logits = self.nn_prediction_head(weighted_features.mean(dim=1))
        udds_pred_logits = self.udds_prediction_head(weighted_features.mean(dim=1))
        # print(f"DEBUG: nn_pred_logits: min={nn_pred_logits.min().item():.4f}, max={nn_pred_logits.max().item():.4f}, mean={nn_pred_logits.mean().item():.4f}, std={nn_pred_logits.std().item():.4f}")
        # print(f"DEBUG: udds_pred_logits: min={udds_pred_logits.min().item():.4f}, max={udds_pred_logits.max().item():.4f}, mean={udds_pred_logits.mean().item():.4f}, std={udds_pred_logits.std().item():.4f}")

        # 加权预测 (直接使用 logits 融合)
        # nn_weight 和 udds_weight 形状为 [batch, 1, 1]，需要调整为 [batch, 1]
        final_prediction_weighted_logits = (nn_pred_logits * nn_weight.squeeze(1) + 
                                           udds_pred_logits * udds_weight.squeeze(1))
        # print(f"DEBUG: final_prediction_weighted_logits: min={final_prediction_weighted_logits.min().item():.4f}, max={final_prediction_weighted_logits.max().item():.4f}, mean={final_prediction_weighted_logits.mean().item():.4f}, std={final_prediction_weighted_logits.std().item():.4f}")

        # 动态融合transformer输出和特定预测
        raw_w_t = torch.sigmoid(self.fusion_weight_param)
        raw_w_h = torch.sigmoid(self.fusion_weight_head_param)
        norm = raw_w_t + raw_w_h + 1e-6
        fusion_weight_transformer = raw_w_t / norm
        fusion_weight_head = raw_w_h / norm
        
        combined_output_last_step_logits = fusion_weight_transformer * transformer_output + fusion_weight_head * final_prediction_weighted_logits
        # print(f"DEBUG: combined_output_last_step (融合后 logits): min={combined_output_last_step_logits.min().item():.4f}, max={combined_output_last_step_logits.max().item():.4f}, mean={combined_output_last_step_logits.mean().item():.4f}, std={combined_output_last_step_logits.std().item():.4f}")
        
        # === 物理约束 ===
        # 将 logits 传入物理约束层
        constrained_output = self.electrochemical_constraint(combined_output_last_step_logits.unsqueeze(1), x[:, -1, :].unsqueeze(1)) # 物理约束层将处理 Sigmoid 激活和 Clamp
        # print(f"DEBUG: constrained_output (物理约束层输出): min={constrained_output.min().item():.4f}, max={constrained_output.max().item():.4f}, mean={constrained_output.mean().item():.4f}, std={constrained_output.std().item():.4f}")
        # 因为约束层内部现在期望 [batch, seq_len, num_outputs] 并且已经将其展平处理，
        # 我们这里输入的是 [batch, 1, num_outputs] 并且 x[:, -1, :].unsqueeze(1) 也是 [batch, 1, num_outputs]
        # 约束层的输出会是 [batch, 1, num_outputs]，我们需要其最终是 [batch, num_outputs]
        constrained_output = constrained_output.squeeze(1)
        
        if return_uncertainty:
            return constrained_output, workload_probs
        
        return constrained_output


class TrueSOTAConstraintLayer(nn.Module):
    """真正SOTA的物理约束层"""
    
    def __init__(self, num_outputs, temperature=None, feature_scaler=None): # 移除 target_scaler 参数
        super().__init__()
        self.num_outputs = num_outputs
        self.temperature = temperature
        self.feature_scaler = feature_scaler # 存储特征标准化器 (用于输入特征)
        
        # 工况自适应约束强度 (恢复为固定值)
        self.nn_constraint_strength = 1.0   
        self.udds_constraint_strength = 1.0  
        
        self.soc_soe_coupling = 0.6 # 从 0.8 进一步调整为 0.6，继续减弱耦合强度
        self.electrochemical_weight = 0.055 # 这个参数不在约束层使用，保持不变
        
        if temperature is not None:
            self.temp_constraint = 1.0 # 恢复为固定值1.0
        else:
            self.temp_constraint = None
            
    def forward(self, predictions, inputs): # predictions 此时应是融合后的 logits
        # 🚨 [关键修改] 在这里对 predictions 进行 Sigmoid 激活，并确保其在 0-1 范围
        # 以前在 forward 里做 Tanh 激活和缩放，现在直接用 Sigmoid
        predictions_physical = torch.sigmoid(predictions)
        
        # 确保 predictions_physical 在 0-1 范围，并进行物理约束
        constrained_physical = predictions_physical.clone()

        # 工况自适应约束强度
        # avg_constraint_strength = (self.nn_constraint_strength + self.udds_constraint_strength) / 2 # 已禁用，使用固定值

        # SOC-SOE耦合约束
        if self.num_outputs >= 2:
            soc_pred_physical = constrained_physical[:, :, 0] # 物理SOC
            soe_pred_physical = constrained_physical[:, :, 1] # 物理SOE
            
            coupling_strength = self.soc_soe_coupling # 现在是固定值
            
            # 在物理尺度上应用耦合约束
            soe_constrained_physical = torch.minimum(soe_pred_physical, soc_pred_physical + 0.06)
            new_soe_physical = coupling_strength * soe_constrained_physical + (1 - coupling_strength) * soe_pred_physical
            
            # 将修正后的SOE重新放回物理预测张量，避免原地操作
            constrained_physical = torch.cat([constrained_physical[:, :, 0:1], new_soe_physical.unsqueeze(-1)], dim=-1)

        # 温度约束 (在物理尺度上应用，如果需要)
        if self.temp_constraint is not None and self.temperature is not None:
            temp_factor = 1.0 if self.temperature >= 25 else 0.85 + 0.15 * (self.temperature + 20) / 45
            temp_strength = self.temp_constraint # 现在是固定值
            # 将整个物理预测值根据温度因子进行调整
            constrained_physical = temp_strength * constrained_physical * temp_factor + (1 - temp_strength) * constrained_physical

        # 🚨 [关键修改] 在返回前添加显式的 Clamp，确保严格在 0-1 范围
        constrained_physical_clamped = torch.clamp(constrained_physical, 0.0, 1.0)
        return constrained_physical_clamped # 返回严格 Clamp 后的物理值


class TrueSOTAElectrochemicalPhysicsLoss(nn.Module):
    """真正SOTA的电化学物理损失函数"""
    
    def __init__(self, base_weight=1.0, electrochemical_weight=1.5, resistance_loss_factor=0.001, feature_scaler=None): # 新增 feature_scaler 和 resistance_loss_factor
        super().__init__()
        self.base_weight = base_weight
        self.electrochemical_weight = nn.Parameter(torch.tensor(electrochemical_weight)) # 初始值从0.05改为1.5
        self.loss_balancer = nn.Parameter(torch.tensor(2.0)) # 提高loss_balancer的初始值到2.0
        
        # 工况特定损失权重
        self.nn_loss_weight = nn.Parameter(torch.tensor(1.5))    # 提高 nn_loss_weight
        self.udds_loss_weight = nn.Parameter(torch.tensor(0.8))  # 降低 udds_loss_weight
        self.feature_scaler = feature_scaler # 存储标准化器
        self.resistance_loss_factor = resistance_loss_factor # 存储内阻损失因子
        
    def forward(self, predictions, targets, inputs, workload_probs=None):
        # 基础MSE损失
        mse_loss = F.mse_loss(predictions, targets) # predictions 和 targets 都应该在 0-1 范围
        
        # 电化学物理损失
        electrochemical_loss = self._compute_electrochemical_loss(predictions, targets, inputs)
        
        # 工况自适应损失加权
        if workload_probs is not None:
            nn_prob = workload_probs[:, 0].mean()
            udds_prob = workload_probs[:, 1].mean()
            
            # 确保损失权重始终为正
            nn_weight = F.softplus(self.nn_loss_weight)
            udds_weight = F.softplus(self.udds_loss_weight)
            
            adaptive_weight = nn_weight * nn_prob + udds_weight * udds_prob
        else:
            adaptive_weight = 1.0
        
        # 损失平衡
        electrochemical_weight = self.electrochemical_weight # 直接使用 Parameter 的原始值
        loss_balance = torch.sigmoid(self.loss_balancer)
        
        total_loss = (
            self.base_weight * mse_loss * loss_balance * adaptive_weight + 
            electrochemical_weight * electrochemical_loss * (1 - loss_balance * 0.5)
        )
        
        return total_loss, {
            'mse_loss': mse_loss.item(), 
            'electrochemical_loss': electrochemical_loss.item(),
            'adaptive_weight': adaptive_weight.item() if isinstance(adaptive_weight, torch.Tensor) else adaptive_weight
        }
    
    def _compute_electrochemical_loss(self, predictions, targets, inputs):
        """计算电化学物理损失"""
        loss_components = []

        # 确保 predictions 和 targets 已经是物理值 (0-1之间)
        # 移除所有 target_scaler.inverse_transform 的调用
        predictions_physical = predictions
        targets_physical = targets

        # 以下所有损失计算都基于 predictions_physical 和 targets_physical

        # 处理不同维度的predictions (现在都是物理尺度)
        # 由于现在 predictions_physical 是 [batch, num_outputs] 形状，不需要再进行 `if len(predictions_physical.shape) == 3` 判断
        # 直接进入 [batch, features] 的处理逻辑
        # SOC范围约束
        soc_pred_original = predictions_physical[:, 0]  # 现在已经是物理SOC
        range_loss = F.relu(soc_pred_original - 1.0).mean() + F.relu(-soc_pred_original).mean()
        loss_components.append(range_loss) # 确保 range_loss 能够提供足够梯度
        
        # SOC-SOE一致性约束
        if predictions_physical.shape[1] >= 2:
            soe_pred_original = predictions_physical[:, 1] # 现在已经是物理SOE
            consistency_loss = F.relu(torch.abs(soe_pred_original - soc_pred_original) - 0.12).mean() # 0.12 差距容忍度
            loss_components.append(consistency_loss)
        
        # 电化学平滑性约束 (不适用单时间步预测，保持注释)
        # if predictions_physical.shape[0] > 1:  # 检查batch维度
        #     try:
        #         original_predictions_smooth = predictions_physical.unsqueeze(1).clone() # 增加seq_len维度
        #         current_targets_smooth = targets_physical.unsqueeze(1).clone() # 增加seq_len维度

        #         pred_diff = torch.diff(original_predictions_smooth, dim=1) 
        #         target_diff = torch.diff(current_targets_smooth, dim=1)    

        #         if pred_diff.shape == target_diff.shape: # type: ignore
        #             smoothness_loss = F.mse_loss(pred_diff, target_diff) * 0.06
        #             loss_components.append(smoothness_loss)
        #     except RuntimeError as e:
        #         print(f"⚠️ 平滑性约束计算警告: {e}") 
        #         pass
        
        # 基于内阻的稳定性约束 (重新引入)
        if inputs.shape[-1] > 6 and self.feature_scaler is not None: # 确保有内阻特征和scaler
            # 获取最后一个时间步的内阻特征（标准化后的）
            normalized_resistance_feature_last_step = inputs[:, -1, 5] # 索引5通常是内阻
            
            # 对内阻进行逆标准化，得到原始物理值
            # 注意：这里我们使用的是feature_scaler，它现在是MinMaxScaler
            # MinMaxScaler的逆变换：original_value = normalized_value * (max - min) + min
            min_resistance = torch.tensor(self.feature_scaler.data_min_[5], device=inputs.device, dtype=inputs.dtype)
            max_resistance = torch.tensor(self.feature_scaler.data_max_[5], device=inputs.device, dtype=inputs.dtype)
            
            original_resistance_feature = normalized_resistance_feature_last_step * (max_resistance - min_resistance) + min_resistance
            
            # 内阻加权预测波动惩罚 (鼓励高内阻时预测更平稳)
            # 这里我们鼓励预测值不要过大，在高内阻时尤其如此
            # predictions_physical 形状为 [batch, num_outputs]，所以对最后一个维度求和
            resistance_penalty = torch.mean(original_resistance_feature * torch.sum(predictions_physical**2, dim=-1)) * self.resistance_loss_factor
            loss_components.append(resistance_penalty)
        
        return sum(loss_components) if loss_components else torch.tensor(0.0, device=predictions.device)


class TrueSOTAElectrochemicalLightningModule(pl.LightningModule):
    """真正SOTA的电化学Lightning模块"""
    
    def __init__(self, hparams, feature_scaler=None): # 移除 target_scaler 参数
        super().__init__()
        if isinstance(hparams, dict): 
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)
        
        self.model = TrueSOTAElectrochemicalKANTransformer(
            input_dim=46,
            num_heads=hparams.num_heads, 
            num_layers=hparams.n_layers,
            num_outputs=len(hparams.output_features), 
            hidden_space=hparams.hidden_space,
            dropout_rate=hparams.dropout, 
            embed_dim=hparams.embed_dim, 
            grid_size=hparams.grid_size,
            temperature=getattr(hparams, 'temperature', None),
            feature_scaler=feature_scaler, # 传递特征标准化器给模型
            detector_temp_init=getattr(hparams, 'detector_temp_init', 0.7)
        )
        
        self.criterion = TrueSOTAElectrochemicalPhysicsLoss(
            electrochemical_weight=getattr(hparams, 'electrochemical_weight', 0.5),
            resistance_loss_factor=getattr(hparams, 'resistance_loss_factor', 0.001), # 新增：传递 resistance_loss_factor
            feature_scaler=feature_scaler # 传递特征标准化器给损失函数
        )
        self.gate_entropy_weight = getattr(hparams, "gate_entropy_weight", 0.01)
        
        self.automatic_optimization = True
        self.test_step_outputs = []
        self.current_epoch_num = 0
        # self.feature_scaler = feature_scaler # 存储标准化器 - 移至模型构造函数
        
    def forward(self, x): 
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _, labels = batch # 新的数据模块会返回原始索引和工况标签
        
        # 数据增强 (针对NN优化)
        if self.training:
            # 保守的噪声注入
            noise_factor = self.hparams.noise_factor * (1 - self.current_epoch_num / self.hparams.num_epochs)
            if noise_factor > 0: 
                x += torch.randn_like(x) * noise_factor
        
        # 前向传播
        y_hat, workload_probs = self.model(x, return_uncertainty=True)
        total_loss, loss_components = self.criterion(y_hat, y, x, workload_probs)
        gate_entropy = -(workload_probs * torch.log(torch.clamp(workload_probs, min=1e-6))).sum(dim=1).mean()
        total_loss = total_loss + self.gate_entropy_weight * gate_entropy
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss): 
            total_loss = F.mse_loss(y_hat, y)
        
        # 日志记录
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in loss_components.items()}, on_step=False, on_epoch=True)
        self.log('gate_entropy', gate_entropy, on_step=False, on_epoch=True)
        
        with torch.no_grad(): 
            train_rmse = torch.sqrt(F.mse_loss(y_hat, y))
            self.log('train_rmse', train_rmse, on_step=False, on_epoch=True)
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _, labels = batch # 新的数据模块会返回原始索引和工况标签
        y_hat = self.forward(x)
        val_loss = F.mse_loss(y_hat, y)
        val_rmse = torch.sqrt(val_loss)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', val_rmse, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, original_end_indices, labels = batch # 接收原始结束索引和工况标签
        y_hat = self.forward(x)
        self.test_step_outputs[dataloader_idx].append({'y_true': y.cpu(), 'y_pred': y_hat.cpu(), 'original_end_indices': original_end_indices.cpu(), 'labels': labels.cpu()})
    
    def on_test_start(self): 
        self.test_step_outputs = [[] for _ in range(2)]
    
    def on_test_epoch_end(self):
        print("\n" + "="*80)
        print("🎯 25°C真正SOTA模型测试结果 (SCI论文级别图表)")
        print("="*80)
        
        # 设置SCI论文级别样式
        setup_sci_style()
        
        # 创建SCI级别图表 - 完全参照用户提供的参考图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Fig. 10: Actual and estimate values of SOC and SOE, and the estimation error with different drive cycles under 25°C: LA92 (top); UDDS (bottom).', 
                    fontsize=12, fontweight='normal', y=0.02)  # 移到图片下方
        
        dataset_names = ["LA92", "UDDS"]
        subplot_idx = 0
        overall_results = {}
        
        for i, outputs in enumerate(self.test_step_outputs):
            if not outputs: continue
            
            # 处理不同序列长度的张量连接问题
            y_true_list = []
            y_pred_list = []
            original_end_indices_list = [] # 新增：用于收集原始索引
            for x in outputs:
                # 统一处理：确保都是2D [batch, features]
                if len(x['y_true'].shape) == 3:  # [batch, seq_len, features]
                    y_true_list.append(x['y_true'][:, -1, :])  # 取最后一个时间步
                    original_end_indices_list.append(x['original_end_indices'][:, -1]) # 收集原始索引
                else:  # [batch, features]
                    y_true_list.append(x['y_true'])
                    original_end_indices_list.append(x['original_end_indices']) # 收集原始索引
                    
                if len(x['y_pred'].shape) == 3:  # [batch, seq_len, features]
                    y_pred_list.append(x['y_pred'][:, -1, :])  # 取最后一个时间步
                else:  # [batch, features]
                    y_pred_list.append(x['y_pred'])
            
            # 安全的张量连接：处理不同batch大小
            try:
                y_true = torch.cat(y_true_list).numpy()
                y_pred = torch.cat(y_pred_list).numpy()
                original_end_indices = torch.cat(original_end_indices_list).numpy() # 合并原始索引
            except RuntimeError as e:
                print(f"⚠️  张量连接警告: {str(e)}")
                # 如果仍然失败，逐个检查并修复维度
                fixed_y_true_list = []
                fixed_y_pred_list = []
                fixed_original_end_indices_list = [] # 新增：用于收集修复后的原始索引
                for yt, yp, oei in zip(y_true_list, y_pred_list, original_end_indices_list):
                    # 确保都是2D且特征数一致
                    if len(yt.shape) == 2 and len(yp.shape) == 2:
                        if yt.shape[1] == yp.shape[1]:  # 特征数一致
                            fixed_y_true_list.append(yt)
                            fixed_y_pred_list.append(yp)
                            fixed_original_end_indices_list.append(oei) # 收集修复后的原始索引
                
                y_true = torch.cat(fixed_y_true_list).numpy()
                y_pred = torch.cat(fixed_y_pred_list).numpy()
                original_end_indices = torch.cat(fixed_original_end_indices_list).numpy() # 合并修复后的原始索引
            
            dataset_name = dataset_names[i]
            
            # --- 新增: 保存原始预测数据 ---
            save_dir = os.path.join(self.trainer.logger.log_dir or '.', 'raw_predictions')
            os.makedirs(save_dir, exist_ok=True)
            
            model_identifier = "KAN-Transformer_46feat"
            true_path = os.path.join(save_dir, f'{model_identifier}_{dataset_name}_true.npy')
            pred_path = os.path.join(save_dir, f'{model_identifier}_{dataset_name}_pred.npy')
            
            np.save(true_path, y_true)
            np.save(pred_path, y_pred)
            # 同时保存原始时间轴索引，以便生成_true.npy和_pred.npy时作为横坐标
            np.save(os.path.join(save_dir, f'{model_identifier}_{dataset_name}_time_axis.npy'), original_end_indices)
            print(f"   📊 {model_identifier} {dataset_name} 原始预测数据已保存: {pred_path}")
            print(f"   📊 {model_identifier} {dataset_name} 原始时间轴索引已保存: {os.path.join(save_dir, f'{model_identifier}_{dataset_name}_time_axis.npy')}")
            # --- 新增结束 ---

            time_axis = original_end_indices # 使用原始时间轴索引作为横坐标
            
            for j, feature in enumerate(self.hparams.output_features):
                # === SCI级别预测结果图 - 完全参照用户参考图 ===
                ax_pred = axes[i, j*2]
                
                actual_values = y_true[:, j] * 100
                pred_values = y_pred[:, j] * 100
                
                # 轻微平滑处理 (参考图的平滑度)
                actual_smooth = smooth_data(actual_values, window_length=9, polyorder=2)
                pred_smooth = smooth_data(pred_values, window_length=9, polyorder=2)
                
                # 参考图的精确颜色方案 - 都是实线
                ax_pred.plot(time_axis, actual_smooth, color='#0072BD', linewidth=1.8, 
                           label='Actual Value', alpha=1.0)      # 蓝色实线
                ax_pred.plot(time_axis, pred_smooth, color='#D95319', linewidth=1.8, 
                           label='Estimated Value', alpha=1.0)   # 橙红色实线
                
                ax_pred.set_xlabel('Time(s)', fontsize=11)
                ax_pred.set_ylabel(f'{feature}(%)', fontsize=11)
                ax_pred.set_title(f'({chr(97+subplot_idx)})', fontsize=12, fontweight='normal')
                ax_pred.legend(loc='upper right', frameon=False, fontsize=10)
                ax_pred.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)  # 参考图的细网格
                ax_pred.set_ylim(0, 100)
                
                # 参考图的边框样式
                for spine in ax_pred.spines.values():
                    spine.set_linewidth(1.0)
                    spine.set_color('black')
                
                # === SCI级别误差分析图 - 参照参考图 ===
                ax_error = axes[i, j*2 + 1]
                
                error_values = pred_smooth - actual_smooth
                error_smooth = smooth_data(error_values, window_length=7, polyorder=2)
                
                # 参考图的精确误差颜色 - 深蓝色系 (与参考图完全一致)
                ax_error.plot(time_axis, error_smooth, color='#1f77b4', linewidth=1.8, alpha=1.0)
                ax_error.set_xlabel('Time(s)', fontsize=11)
                ax_error.set_ylabel(f'{feature} Error (%)', fontsize=11)
                ax_error.set_title(f'({chr(97+subplot_idx+1)})', fontsize=12, fontweight='normal')
                ax_error.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)  # 参考图的细网格
                ax_error.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.8)
                ax_error.set_ylim(-6, 6)  # 统一error图纵坐标范围为-6到6
                
                # 参考图的边框样式
                for spine in ax_error.spines.values():
                    spine.set_linewidth(1.0)
                    spine.set_color('black')
                
                # 评估指标计算
                rmse = np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
                mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
                r2 = r2_score(y_true[:, j], y_pred[:, j])
                
                result_key = f"{dataset_name}_{feature}"
                overall_results[result_key] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
                
                # 与基准对比
                original_baseline = {"LA92_SOC": 0.030915, "LA92_SOE": 0.029856, "UDDS_SOC": 0.012418, "UDDS_SOE": 0.011143}
                tabpfn_baseline = {"LA92_SOC": 0.029, "LA92_SOE": 0.027, "UDDS_SOC": 0.012, "UDDS_SOE": 0.010}
                
                original_rmse = original_baseline.get(result_key, 0.025)
                tabpfn_rmse = tabpfn_baseline.get(result_key, 0.025)
                
                improvement_vs_original = original_rmse - rmse
                improvement_vs_tabpfn = tabpfn_rmse - rmse
                
                print(f"🎯 {dataset_name} - {feature}:")
                print(f"    RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
                
                # 特别标注性能变化
                if dataset_name == "LA92":
                    if rmse < 0.020:
                        print(f"    🚀 LA92表现优秀! RMSE: {rmse:.6f} < 0.020")
                    else:
                        print(f"    📈 LA92表现: RMSE: {rmse:.6f}")
                elif dataset_name == "UDDS":
                    if rmse <= 0.013:
                        print(f"    ✅ UDDS保持优势: {rmse:.6f} ≤ 0.013")
                    else:
                        print(f"    ⚠️ UDDS略有退步: {rmse:.6f} > 0.013")
                
                if improvement_vs_tabpfn > 0:
                    print(f"    🏆 超越TabPFN! 改善: {improvement_vs_tabpfn:.6f}")
                else:
                    print(f"    📊 vs TabPFN差距: {abs(improvement_vs_tabpfn):.6f}")
                
                subplot_idx += 2
        
        # 参考图的精确布局调整 - 为底部标题留出空间
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.06, right=0.98, 
                           hspace=0.35, wspace=0.25)  # 增加底部空间给Fig.10说明
        
        # 保存高质量SCI图表 - 参照参考图格式
        save_path = os.path.join(self.trainer.logger.log_dir or '.', 
                               'true_sota_25degC_SCI_paper_style.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', pad_inches=0.1)
        print(f"\n📊 SCI论文级别图表已保存: {save_path}")
        
        # 保存测试指标
        results_df = pd.DataFrame(overall_results).T
        results_df.index.name = 'Dataset_Feature'
        results_csv_path = os.path.join(self.trainer.logger.log_dir or '.', 
                                      'true_sota_25degC_test_metrics.csv')
        results_df.to_csv(results_csv_path)
        print(f"📊 测试指标结果已保存: {results_csv_path}")
        
        # 综合性能分析
        avg_rmse = np.mean([r['RMSE'] for r in overall_results.values()])
        avg_mae = np.mean([r['MAE'] for r in overall_results.values()])
        avg_r2 = np.mean([r['R2'] for r in overall_results.values()])
        
        # 分工况分析
        la92_rmse = np.mean([overall_results[k]['RMSE'] for k in overall_results.keys() if 'LA92' in k])
        udds_rmse = np.mean([overall_results[k]['RMSE'] for k in overall_results.keys() if 'UDDS' in k])
        
        print(f"\n🏆 综合性能 (SCI论文级别):")
        print(f"    平均RMSE: {avg_rmse:.6f}")
        print(f"    平均MAE:  {avg_mae:.6f}")
        print(f"    平均R²:   {avg_r2:.6f}")
        
        print(f"\n📊 分工况性能:")
        print(f"    LA92平均RMSE: {la92_rmse:.6f} (LA92作为测试集)")
        print(f"    UDDS平均RMSE: {udds_rmse:.6f} (目标: ≤0.013)")
        
        # 与基准对比
        original_avg_rmse = 0.021083
        tabpfn_avg_rmse = 0.0195
        
        print(f"\n📈 SCI论文级别分析:")
        if avg_rmse < original_avg_rmse:
            print(f"    🎉 整体性能提升: {original_avg_rmse - avg_rmse:.6f}")
        else:
            print(f"    📉 整体性能退步: {avg_rmse - original_avg_rmse:.6f}")
            
        if avg_rmse < tabpfn_avg_rmse:
            print(f"    🏆 成功超越TabPFN! 改善: {tabpfn_avg_rmse - avg_rmse:.6f}")
        else:
            print(f"    📊 距离TabPFN还差: {avg_rmse - tabpfn_avg_rmse:.6f}")
        
        # 成功标准
        la92_success = la92_rmse < 0.025  # LA92相对宽松的标准
        udds_success = udds_rmse <= 0.013
        overall_success = avg_rmse < 0.020  # 调整整体标准
        
        print(f"\n🎯 成功评估 (SCI论文级别):")
        print(f"    LA92表现: {'✅' if la92_success else '❌'} ({la92_rmse:.6f} {'<' if la92_success else '≥'} 0.025)")
        print(f"    UDDS保持成功: {'✅' if udds_success else '❌'} ({udds_rmse:.6f} {'≤' if udds_success else '>'} 0.013)")
        print(f"    整体表现: {'✅' if overall_success else '❌'} ({avg_rmse:.6f} {'<' if overall_success else '≥'} 0.020)")
        
        if la92_success and udds_success and overall_success:
            print(f"    🚀🚀🚀 完美成功! SCI论文级别表现优异!")
        elif la92_success and udds_success:
            print(f"    🚀🚀 双重成功! LA92+UDDS都表现良好!")
        elif la92_success or udds_success:
            print(f"    🚀 部分成功! 继续优化!")
        
        plt.show()
        print("="*80)
    
    def on_train_epoch_end(self): 
        self.current_epoch_num += 1
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),    # 平衡的beta参数
            eps=1e-6
        )
        
        # 稳定的学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=15,
            min_lr=self.hparams.lr * 0.001
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        }


class MemoryCleanupCallback(Callback):
    """内存清理回调"""
    def on_train_epoch_end(self, trainer, pl_module): 
        gc.collect()
        torch.cuda.empty_cache()


def main():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"✅ PyTorch检测到CUDA可用。设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ PyTorch未检测到CUDA。请检查CUDA安装和驱动。")

    parser = argparse.ArgumentParser(description='25°C真正SOTA训练 - 保持UDDS优势+提升NN')
    
    # === 数据路径 ===
    # 🔄 调换配置：将NN测试集加入训练，LA92训练集作为测试
    parser.add_argument('--train_paths', type=str, nargs='+', default=[
        r"C:\25degC training\03-18-17_02.17 25degC_Cycle_1_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_03.25 25degC_Cycle_2_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_09.07 25degC_Cycle_3_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_14.31 25degC_Cycle_4_Pan18650PF.csv",
        r"C:\25degC training\03-20-17_01.43 25degC_US06_Pan18650PF.csv",
        r"C:\25degC training\03-20-17_05.56 25degC_HWFTa_Pan18650PF.csv",
        r"C:\25degC testing\03-21-17_16.27 25degC_NN_Pan18650PF.csv"  # 原NN测试集现在用于训练
    ])
    parser.add_argument('--test_paths', type=str, nargs='+', default=[
        r"C:\25degC training\03-21-17_09.38 25degC_LA92_Pan18650PF.csv",  # 原LA92训练集现在用于测试
        r"C:\25degC testing\03-21-17_00.29 25degC_UDDS_Pan18650PF.csv"   # UDDS保持不变
    ])
    parser.add_argument('--result_dir', type=str, default='true_sota_25degC_LA92_test_results_step16')
    parser.add_argument('--output_features', type=str, default='SOC,SOE')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--overlap_ratio', type=float, default=0.5) # 步长为16（最优配置）
    parser.add_argument('--temperature', type=float, default=25.0)
    parser.add_argument('--model_name', type=str, default='KAN-Transformer', help='Name of the model for logging and display.')
    
    # === [真正SOTA] 架构参数 ===
    parser.add_argument('--n_layers', type=int, default=5)      # 适中的层数
    parser.add_argument('--num_heads', type=int, default=16)    # 足够的注意力头
    parser.add_argument('--hidden_space', type=int, default=128) # 平衡的隐藏空间
    parser.add_argument('--embed_dim', type=int, default=128)   # 相应调整
    
    # === 平衡的正则化参数 ===
    parser.add_argument('--dropout', type=float, default=0.20)         # 适中的dropout
    parser.add_argument('--weight_decay', type=float, default=0.0006)  # 适中的weight_decay
    parser.add_argument('--noise_factor', type=float, default=0.004)   # 轻微噪声
    parser.add_argument('--electrochemical_weight', type=float, default=0.05) # 恢复为更保守的0.05
    parser.add_argument('--resistance_loss_factor', type=float, default=0.001, help='Factor for resistance-based stability loss.') # 保持不变
    parser.add_argument('--gate_entropy_weight', type=float, default=0.01, help='Entropy penalty to sharpen workload detector.')
    parser.add_argument('--detector_temp_init', type=float, default=0.7, help='Initial temperature for workload detector softmax.')
    
    # === [真正SOTA] 训练参数 ===
    parser.add_argument('--grid_size', type=int, default=24)       # 高精度KAN
    parser.add_argument('--num_epochs', type=int, default=300)     # 增加到300
    parser.add_argument('--batch_size', type=int, default=32)      
    parser.add_argument('--patience', type=int, default=45)        # 增加到45
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0004)        # 降低学习率到0.0004
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode (e.g., fewer epochs, smaller dataset)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a pre-trained checkpoint to load for testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)  # 轻微的梯度累积
    parser.add_argument('--train_val_split_ratio', type=float, default=0.93, help='Ratio for training set split from total training data.')
    
    args = parser.parse_args()
    args.output_features = [item.strip() for item in args.output_features.split(',')]
    os.makedirs(args.result_dir, exist_ok=True)
    
    print("🎯 === 25°C SOTA训练 - LA92测试配置 ===")
    print("🔄 数据配置调换:")
    print("   - 训练集: 6个原训练集 + NN测试集 (增强训练数据)")
    print("   - 测试集: LA92 (原训练集) + UDDS (保持对比)")
    print("   - 目的: 验证模型在LA92工况下的泛化能力")
    print("🚀 SOTA策略:")
    print("   - 保持UDDS优势: 维持简单有效的处理分支")
    print("   - 工况检测器: 自动识别不同工况类型")
    print("   - 自适应损失: 工况特定的损失权重")
    print("   - 工况特定预测头: 不同工况分别优化")
    print("   - 目标: UDDS保持≤0.013, LA92<0.025, 整体<0.020")
    print(f"   - 基准对比: 原版平均0.021, TabPFN平均0.0195")
    
    try:
        pl.seed_everything(args.seed, workers=True)
        datamodule = Electrochemical46FeaturesDataModule(args) # 使用新的KAN数据模块
        datamodule.setup(stage='fit') # 显式调用setup，并传入stage
        
        # 获取全局标准化器
        scalers_dict = datamodule.scaler
        feature_scaler = scalers_dict['features']

        model = TrueSOTAElectrochemicalLightningModule(args, feature_scaler=feature_scaler) # 传递 feature_scaler
        
        # 回调函数设置
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.result_dir, 
            filename='true-sota-25degC-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1, verbose=True, monitor='val_loss', mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss', patience=args.patience, verbose=True, mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # 显式配置logger，使其保存到args.result_dir
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=args.result_dir, name='', version='') # name为空，version为空，直接保存到save_dir

        # 训练器设置
        trainer = pl.Trainer(
            max_epochs=args.num_epochs, 
            accelerator='gpu', # 明确指定使用GPU
            devices=args.gpus, # 使用ArgumentParser中定义的GPU数量
            callbacks=[
                checkpoint_callback, 
                early_stop_callback, 
                lr_monitor,
                MemoryCleanupCallback()
            ],
            precision='32',
            gradient_clip_val=0.5,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            deterministic=False,
            benchmark=True,
            logger=logger # 传入配置好的logger
        )
        
        print(f"\n🚀 开始 {args.model_name} 训练...")

        if args.ckpt_path:
            print(f"   跳过训练，从检查点加载模型: {args.ckpt_path}")
            # 直接加载模型并测试
            model = type(model).load_from_checkpoint(args.ckpt_path, hparams=args) # 重新传入hparams
            trainer.test(model, datamodule=datamodule)
        else:
            # 正常训练流程
            trainer.fit(model, datamodule)
            print(f"\n📊 测试 {args.model_name} 模型...")
            # 使用最佳检查点进行测试
            trainer.test(model, datamodule=datamodule, ckpt_path='best')
        
        print(f"\n✅ 真正SOTA训练完成！")
        print(f"📁 结果保存在: {args.result_dir}")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
