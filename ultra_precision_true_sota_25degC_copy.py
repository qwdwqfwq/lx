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
                 temperature=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.temperature = temperature
        
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
        total_encoded_dim = hidden_space//4 * 3 + hidden_space//6 * 2
        self.workload_detector = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_space//2),
            nn.LayerNorm(hidden_space//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space//2, hidden_space//4),
            nn.ReLU(),
            nn.Linear(hidden_space//4, 2),  # NN vs UDDS
            nn.Softmax(dim=-1)
        )
        
        # === [策略3] 工况特定的处理分支 ===
        
        # UDDS分支 (保持简单有效的设计)
        self.udds_branch = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_space),
            nn.LayerNorm(hidden_space),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.15)
        )
        
        # NN分支 (专门针对NN的复杂设计)
        self.nn_branch = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_space + 32),  # 更大容量
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
        self.electrochemical_constraint = TrueSOTAConstraintLayer(num_outputs, temperature)
        
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
        
        # === 工况检测 ===
        workload_probs = self.workload_detector(concatenated_features.mean(dim=1))  # [batch, 2]
        nn_prob = workload_probs[:, 0]      # NN工况概率
        udds_prob = workload_probs[:, 1]    # UDDS工况概率
        
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
        nn_weight = nn_prob.unsqueeze(1).unsqueeze(2) # [batch, 1, 1]
        udds_weight = udds_prob.unsqueeze(1).unsqueeze(2) # [batch, 1, 1]
        
        weighted_features = nn_enhanced * nn_weight + udds_enhanced * udds_weight
        
        # === 主要Transformer处理 (现在只输出最后一个时间步) ===
        # 修正：self.main_transformer 已输出 [batch, num_outputs] 形状，无需再取[:, -1, :]
        transformer_output = self.main_transformer(weighted_features) # [batch, num_outputs]
        
        # === 工况特定预测 (输出单点预测) ===
        nn_pred = self.nn_prediction_head(weighted_features.mean(dim=1)) # [batch, num_outputs]
        udds_pred = self.udds_prediction_head(weighted_features.mean(dim=1)) # [batch, num_outputs]
        
        # 加权预测 (使用 [batch, 1] 权重与 [batch, num_outputs] 预测进行广播)
        final_prediction = (nn_pred * nn_weight.squeeze(1) + 
                          udds_pred * udds_weight.squeeze(1)) # [batch, num_outputs]
        
        # 融合transformer的单点输出和特定预测头输出
        combined_output = 0.6 * transformer_output + 0.4 * final_prediction # [batch, num_outputs]
        
        # === 物理约束 ===
        # 约束层期望 [batch, seq_len, features]，所以将单点输出扩展为 seq_len=1，然后压缩回去
        constrained_output = self.electrochemical_constraint(combined_output.unsqueeze(1), x[:, -1, :].unsqueeze(1))
        constrained_output = constrained_output.squeeze(1) # 最终输出是 [batch, num_outputs]
        
        if return_uncertainty: 
            return constrained_output, workload_probs
        
        return constrained_output


class TrueSOTAConstraintLayer(nn.Module):
    """真正SOTA的物理约束层"""
    
    def __init__(self, num_outputs, temperature=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.temperature = temperature
        
        # 工况自适应约束强度
        self.nn_constraint_strength = nn.Parameter(torch.tensor(0.12))    # NN需要更强约束
        self.udds_constraint_strength = nn.Parameter(torch.tensor(0.08))  # UDDS约束适中
        
        self.soc_soe_coupling = nn.Parameter(torch.tensor(0.10))
        self.electrochemical_weight = nn.Parameter(torch.tensor(0.055))
        
        if temperature is not None:
            self.temp_constraint = nn.Parameter(torch.tensor(0.035))
        else:
            self.temp_constraint = None
            
    def forward(self, predictions, inputs):
        # 基础约束
        constrained = torch.sigmoid(predictions)
        
        # 工况自适应约束 (这里简化处理，实际可以用工况检测结果)
        # 假设batch内混合了NN和UDDS，使用平均约束强度
        avg_constraint_strength = (self.nn_constraint_strength + self.udds_constraint_strength) / 2
        
        # 处理维度：如果是3D张量 [batch, seq_len, features]，需要在最后一个维度操作
        original_shape = constrained.shape
        
        # SOC-SOE耦合约束
        if self.num_outputs >= 2:
            if len(original_shape) == 3:  # [batch, seq_len, features]
                soc_pred = constrained[:, :, 0]  # [batch, seq_len]
                soe_pred = constrained[:, :, 1]  # [batch, seq_len]
            else:  # [batch, features]
                soc_pred = constrained[:, 0]
                soe_pred = constrained[:, 1]
            
            coupling_strength = torch.sigmoid(self.soc_soe_coupling)
            soe_constrained = torch.minimum(soe_pred, soc_pred + 0.06)
            new_soe = coupling_strength * soe_constrained + (1 - coupling_strength) * soe_pred
            
            if len(original_shape) == 3:
                constrained = torch.stack([soc_pred, new_soe], dim=-1)  # [batch, seq_len, 2]
            else:
                constrained = torch.stack([soc_pred, new_soe], dim=-1)  # [batch, 2]
        
        # 温度约束
        if self.temp_constraint is not None and self.temperature is not None:
            temp_factor = 1.0 if self.temperature >= 25 else 0.85 + 0.15 * (self.temperature + 20) / 45
            temp_strength = torch.sigmoid(self.temp_constraint)
            constrained = temp_strength * constrained * temp_factor + (1 - temp_strength) * constrained
        
        # 应用约束强度
        strength = torch.sigmoid(avg_constraint_strength)
        final_output = strength * constrained + (1 - strength) * torch.sigmoid(predictions)
        
        return final_output


class TrueSOTAElectrochemicalPhysicsLoss(nn.Module):
    """真正SOTA的电化学物理损失函数"""
    
    def __init__(self, base_weight=1.0, electrochemical_weight=0.025):
        super().__init__()
        self.base_weight = base_weight
        self.electrochemical_weight = nn.Parameter(torch.tensor(electrochemical_weight))
        self.loss_balancer = nn.Parameter(torch.tensor(1.08))
        
        # 工况特定损失权重
        self.nn_loss_weight = nn.Parameter(torch.tensor(1.2))    # NN需要更大权重
        self.udds_loss_weight = nn.Parameter(torch.tensor(0.8))  # UDDS保持较小权重
        
    def forward(self, predictions, targets, inputs, workload_probs=None):
        # 基础MSE损失
        mse_loss = F.mse_loss(predictions, targets)
        
        # 电化学物理损失
        electrochemical_loss = self._compute_electrochemical_loss(predictions, targets, inputs)
        
        # 工况自适应损失加权
        if workload_probs is not None:
            nn_prob = workload_probs[:, 0].mean()
            udds_prob = workload_probs[:, 1].mean()
            
            # 动态调整损失权重
            nn_weight = torch.sigmoid(self.nn_loss_weight)
            udds_weight = torch.sigmoid(self.udds_loss_weight)
            
            adaptive_weight = nn_weight * nn_prob + udds_weight * udds_prob
        else:
            adaptive_weight = 1.0
        
        # 损失平衡
        electrochemical_weight = torch.sigmoid(self.electrochemical_weight) * 0.05
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
        
        # 处理不同维度的predictions
        if len(predictions.shape) == 3:  # [batch, seq_len, features]
            # SOC范围约束
            soc_pred = predictions[:, :, 0]  # [batch, seq_len]
            range_loss = F.relu(soc_pred - 1.0).mean() + F.relu(-soc_pred).mean()
            loss_components.append(range_loss)
            
            # SOC-SOE一致性约束
            if predictions.shape[2] >= 2:
                soc_pred = predictions[:, :, 0]  # [batch, seq_len]
                soe_pred = predictions[:, :, 1]  # [batch, seq_len]
                consistency_loss = F.relu(torch.abs(soe_pred - soc_pred) - 0.12).mean()
                loss_components.append(consistency_loss)
        else:  # [batch, features]
            # SOC范围约束
            soc_pred = predictions[:, 0]
            range_loss = F.relu(soc_pred - 1.0).mean() + F.relu(-soc_pred).mean()
            loss_components.append(range_loss)
            
            # SOC-SOE一致性约束
            if predictions.shape[1] >= 2:
                soc_pred = predictions[:, 0]
                soe_pred = predictions[:, 1]
                consistency_loss = F.relu(torch.abs(soe_pred - soc_pred) - 0.12).mean()
                loss_components.append(consistency_loss)
        
        # 电化学平滑性约束 (不适用单点预测，故注释掉)
        # if predictions.shape[0] > 1:  # 检查batch维度
        #     try:
        #         pred_diff = torch.diff(predictions, dim=0)
        #         target_diff = torch.diff(targets, dim=0)
        #         # 确保维度匹配
        #         if pred_diff.shape == target_diff.shape:
        #             smoothness_loss = F.mse_loss(pred_diff, target_diff) * 0.06
        #             loss_components.append(smoothness_loss)
        #     except RuntimeError:
        #         # 如果维度不匹配，跳过平滑性约束
        #         pass
        
        # 基于内阻的稳定性约束
        if inputs.shape[-1] > 6:
            resistance_feature = inputs[:, :, 6].mean(dim=1)
            resistance_penalty = torch.mean(resistance_feature * torch.var(predictions, dim=0).sum()) * 0.01
            loss_components.append(resistance_penalty)
        
        return sum(loss_components) if loss_components else torch.tensor(0.0, device=predictions.device)


class TrueSOTAElectrochemicalLightningModule(pl.LightningModule):
    """真正SOTA的电化学Lightning模块"""
    
    def __init__(self, hparams):
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
            temperature=getattr(hparams, 'temperature', None)
        )
        
        self.criterion = TrueSOTAElectrochemicalPhysicsLoss(
            electrochemical_weight=getattr(hparams, 'electrochemical_weight', 0.025)
        )
        
        self.automatic_optimization = True
        self.test_step_outputs = []
        self.current_epoch_num = 0
        
    def forward(self, x): 
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # 适配模型输出为单点预测，目标y也只取最后一个时间步进行损失计算
        # 修正：y已是 [batch, num_outputs] 形状，无需再取[:, -1, :]
        # y = y[:, -1, :]
        
        # 数据增强 (针对NN优化)
        if self.training:
            # 保守的噪声注入
            noise_factor = self.hparams.noise_factor * (1 - self.current_epoch_num / self.hparams.num_epochs)
            if noise_factor > 0: 
                x += torch.randn_like(x) * noise_factor
        
        # 前向传播
        y_hat, workload_probs = self.model(x, return_uncertainty=True)
        total_loss, loss_components = self.criterion(y_hat, y, x, workload_probs)
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss): 
            total_loss = F.mse_loss(y_hat, y)
        
        # 日志记录
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in loss_components.items()}, on_step=False, on_epoch=True)
        
        with torch.no_grad(): 
            train_rmse = torch.sqrt(F.mse_loss(y_hat, y))
            self.log('train_rmse', train_rmse, on_step=False, on_epoch=True)
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # 适配模型输出为单点预测，目标y也只取最后一个时间步
        # 修正：y已是 [batch, num_outputs] 形状，无需再取[:, -1, :]
        # y = y[:, -1, :]
        y_hat = self.forward(x)
        val_loss = F.mse_loss(y_hat, y)
        val_rmse = torch.sqrt(val_loss)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', val_rmse, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        self.test_step_outputs[dataloader_idx].append({'y_true': y.cpu(), 'y_pred': y_hat.cpu()})
    
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
            for x in outputs:
                # 统一处理：确保都是2D [batch, features]
                if len(x['y_true'].shape) == 3:  # [batch, seq_len, features]
                    y_true_list.append(x['y_true'][:, -1, :])  # 取最后一个时间步
                else:  # [batch, features]
                    y_true_list.append(x['y_true'])
                    
                if len(x['y_pred'].shape) == 3:  # [batch, seq_len, features]
                    y_pred_list.append(x['y_pred'][:, -1, :])  # 取最后一个时间步
                else:  # [batch, features]
                    y_pred_list.append(x['y_pred'])
            
            # 安全的张量连接：处理不同batch大小
            try:
                y_true = torch.cat(y_true_list).numpy()
                y_pred = torch.cat(y_pred_list).numpy()
            except RuntimeError as e:
                print(f"⚠️  张量连接警告: {str(e)}")
                # 如果仍然失败，逐个检查并修复维度
                fixed_y_true_list = []
                fixed_y_pred_list = []
                for yt, yp in zip(y_true_list, y_pred_list):
                    # 确保都是2D且特征数一致
                    if len(yt.shape) == 2 and len(yp.shape) == 2:
                        if yt.shape[1] == yp.shape[1]:  # 特征数一致
                            fixed_y_true_list.append(yt)
                            fixed_y_pred_list.append(yp)
                
                y_true = torch.cat(fixed_y_true_list).numpy()
                y_pred = torch.cat(fixed_y_pred_list).numpy()
            
            dataset_name = dataset_names[i]
            time_axis = np.arange(len(y_true))
            
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
        save_path = os.path.join(self.trainer.default_root_dir, 
                               'true_sota_25degC_SCI_paper_style.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', pad_inches=0.1)
        print(f"\n📊 SCI论文级别图表已保存: {save_path}")
        
        # 保存测试指标
        results_df = pd.DataFrame(overall_results).T
        results_df.index.name = 'Dataset_Feature'
        results_csv_path = os.path.join(self.trainer.default_root_dir, 
                                      'true_sota_25degC_test_metrics.csv')
        results_df.to_csv(results_csv_path)
        print(f"📊 测试指标结果已保存: {results_csv_path}")

        # === 保存预测序列、真实值和时间戳 (按数据集和特征单独保存) ===
        for i, outputs in enumerate(self.test_step_outputs):
            if not outputs: continue
            
            y_true_list = []
            y_pred_list = []
            for x in outputs:
                if len(x['y_true'].shape) == 3:
                    y_true_list.append(x['y_true'][:, -1, :])
                else:
                    y_true_list.append(x['y_true'])
                    
                if len(x['y_pred'].shape) == 3:
                    y_pred_list.append(x['y_pred'][:, -1, :])
                else:
                    y_pred_list.append(x['y_pred'])

            y_true_concat = torch.cat(y_true_list).numpy()
            y_pred_concat = torch.cat(y_pred_list).numpy()
            time_axis_concat = np.arange(len(y_true_concat))
            
            dataset_name = dataset_names[i]
            for j, feature in enumerate(self.hparams.output_features):
                df_predictions = pd.DataFrame({
                    'Time(s)': time_axis_concat,
                    f'{feature}_Actual': y_true_concat[:, j],
                    f'{feature}_Predicted': y_pred_concat[:, j],
                })
                prediction_csv_path = os.path.join(
                    self.trainer.default_root_dir, 
                    f'{dataset_name}_{feature}_predictions.csv'
                )
                df_predictions.to_csv(prediction_csv_path, index=False)
                print(f"📊 {dataset_name} - {feature} 预测序列、真实值和时间戳已保存: {prediction_csv_path}")
        
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


class ElectrochemicalDataModule(pl.LightningDataModule):
    """电化学数据模块"""
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.feature_engineer = None
        
    def setup(self, stage=None):
        print(f"🧪 开始电化学特征工程 ({self.hparams.temperature}°C)")
        
        # 使用新的电化学特征数据集
        train_features, train_targets, test_datasets, self.feature_engineer = create_electrochemical_dataset(
            self.hparams.train_paths, 
            self.hparams.test_paths,
            self.hparams.temperature,
            window_size=self.hparams.window_size,
            overlap_ratio=0.5
        )
        
        # 确保特征维度正确
        assert train_features.shape[-1] == 46, f"期望46个特征，实际得到{train_features.shape[-1]}个"
        self.hparams.input_dim = 46
        
        # 数据转换
        X_train_tensor = torch.from_numpy(train_features).float()
        y_train_tensor = torch.from_numpy(train_targets).float()
        
        # 数据划分
        dataset_size = len(X_train_tensor)
        train_size = int(0.93 * dataset_size)  # 93% 训练
        val_size = dataset_size - train_size   # 7% 验证
        print(f"📊 数据划分 (93%/7%): 训练={train_size:,}, 验证={val_size:,}")
        
        full_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 测试数据集处理
        self.test_datasets = []
        for test_dict in test_datasets:
            test_features = test_dict['features']
            test_targets = test_dict['targets']
            test_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(test_features).float(),
                torch.from_numpy(test_targets).float()
            )
            self.test_datasets.append(test_dataset)
            
        print(f"✅ 电化学数据准备完成 (46个科学特征)")
    
    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self): 
        return [torch.utils.data.DataLoader(
            ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        ) for ds in self.test_datasets]


class MemoryCleanupCallback(Callback):
    """内存清理回调"""
    def on_train_epoch_end(self, trainer, pl_module): 
        gc.collect()
        torch.cuda.empty_cache()


def main():
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
    parser.add_argument('--result_dir', type=str, default='true_sota_25degC_LA92_test_results_step12')
    parser.add_argument('--output_features', type=str, default='SOC,SOE')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=25.0)
    
    # === [真正SOTA] 架构参数 ===
    parser.add_argument('--n_layers', type=int, default=5)      # 适中的层数
    parser.add_argument('--num_heads', type=int, default=16)    # 足够的注意力头
    parser.add_argument('--hidden_space', type=int, default=128) # 平衡的隐藏空间
    parser.add_argument('--embed_dim', type=int, default=128)   # 相应调整
    
    # === 平衡的正则化参数 ===
    parser.add_argument('--dropout', type=float, default=0.20)         # 适中的dropout
    parser.add_argument('--weight_decay', type=float, default=0.0006)  # 适中的weight_decay
    parser.add_argument('--noise_factor', type=float, default=0.004)   # 轻微噪声
    parser.add_argument('--electrochemical_weight', type=float, default=0.025) # 平衡的物理权重
    
    # === [真正SOTA] 训练参数 ===
    parser.add_argument('--grid_size', type=int, default=24)       # 高精度KAN
    parser.add_argument('--num_epochs', type=int, default=200)     # 200个周期训练
    parser.add_argument('--batch_size', type=int, default=32)      # 稳定的批次大小
    parser.add_argument('--patience', type=int, default=30)        # 适中的patience
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0006)        # 平衡的学习率
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)  # 轻微的梯度累积
    parser.add_argument('--ckpt_path', type=str, default=None, help='Load model from checkpoint path') # 新增参数

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
        datamodule = ElectrochemicalDataModule(args)
        datamodule.setup()
        
        if args.ckpt_path: # 如果提供了检查点路径，则直接从 {args.ckpt_path} 加载模型进行测试，跳过训练
            print(f"\n🚀 检测到ckpt_path，直接从 {args.ckpt_path} 加载模型进行测试，跳过训练。")
            model = TrueSOTAElectrochemicalLightningModule.load_from_checkpoint(checkpoint_path=args.ckpt_path, hparams=args)
        else:
            model = TrueSOTAElectrochemicalLightningModule(args)
            print(f"\n🚀 开始25°C环境下训练...")
        
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
        
        # 训练器设置
        trainer = pl.Trainer(
            max_epochs=args.num_epochs, 
            accelerator='auto', 
            devices=1,
            logger=False, # 禁用默认的logger，确保所有内容都保存在default_root_dir
            default_root_dir=args.result_dir, # 将默认根目录设置为args.result_dir
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
            benchmark=True
        )
        
        if not args.ckpt_path: # 如果没有提供检查点路径，才执行训练
            print(f"\n🚀 开始25°C环境下训练...")
            trainer.fit(model, datamodule)
        
        print(f"\n📊 测试25°C环境下模型...")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path if args.ckpt_path else 'best')
        
        print(f"\n✅ 真正SOTA训练完成！")
        print(f"📁 结果保存在: {args.result_dir}")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
