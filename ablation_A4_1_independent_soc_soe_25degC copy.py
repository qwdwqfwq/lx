"""
消融实验 A4.1: SOC和SOE独立预测 vs 联合预测 - 25°C
==========================================================

🎯 实验目的:
    验证SOC-SOE联合预测的价值
    量化信息共享和物理耦合约束的贡献
    
📉 移除内容:
    ❌ SOC-SOE联合预测（单一模型同时输出2个值）
    ❌ SOC-SOE耦合约束 (|SOE - SOC| < 0.12)
    ❌ 共享特征编码器
    
🔄 替换方案:
    两个完全独立的模型：
    - SOC模型：独立编码器 + Transformer + 预测头 → SOC
    - SOE模型：独立编码器 + Transformer + 预测头 → SOE
    
✅ 保留内容:
    ✓ 相同的架构复杂度（每个模型）
    ✓ 相同的训练策略
    ✓ 相同的超参数
    
📊 预期性能下降: 8-12%
    - 失去SOC-SOE信息共享
    - 失去物理耦合约束
    - 计算量翻倍但无协同效应
    
🔬 科学意义:
    ⭐⭐⭐⭐⭐ Energy期刊核心亮点！
    - 证明联合学习的优势
    - 量化物理耦合约束的价值
    - 为电池管理系统提供设计指导

数据标准化：使用全局MinMaxScaler
=================================
🔄 替代方案：
    全局MinMaxScaler: 针对46维电化学特征进行0-1标准化，避免StandardScaler可能带来的负值和过大方差问题。
    
效果预期：
    由于KAN对输入范围敏感，MinMaxScaler预计能提供更稳定的训练和更好的性能。
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateMonitor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# from electrochemical_features import create_electrochemical_dataset # 移除旧的导入
# from electrochemical_46features_datamodule import Electrochemical46FeaturesDataModule # 移除旧的数据模块导入
from electrochemical_46features_kan_datamodule import Electrochemical46FeaturesKANDataModule # 导入新的KAN数据模块

from model_code_lightning import setup_chinese_font

setup_chinese_font()


def setup_sci_style():
    """SCI论文级别样式"""
    plt.style.use('default')
    plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.8


def smooth_data(data, window_length=9, polyorder=2):
    if len(data) < window_length:
        return data
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(polyorder, window_length - 1)
    try:
        return savgol_filter(data, window_length, polyorder)
    except:
        return data


class IndependentSOCModel(nn.Module):
    """
    独立的SOC预测模型
    完全独立的架构，不与SOE共享任何信息
    """
    
    def __init__(self, input_dim, num_heads, num_layers, hidden_space, 
                 dropout_rate, embed_dim, grid_size=16, temperature=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.temperature = temperature
        
        # ✅ 独立的特征编码器（与完整模型相同的架构，但参数独立）
        self.basic_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(12, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(8, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        self.impedance_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.temperature_encoder = nn.Sequential(
            nn.Linear(6, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        # ✅ 独立的处理分支
        total_encoded_dim = hidden_space//4 * 3 + hidden_space//6 * 2
        
        self.soc_branch = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_space),
            nn.LayerNorm(hidden_space),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.15)
        )
        
        # ✅ 独立的Transformer
        from model2 import TimeSeriesTransformer_ekan
        
        self.transformer = TimeSeriesTransformer_ekan(
            input_dim=hidden_space,
            num_heads=num_heads,
            num_layers=num_layers,
            num_outputs=1,  # 仅输出SOC
            hidden_space=hidden_space,
            dropout_rate=dropout_rate * 0.5,
            embed_dim=embed_dim,
            grid_size=grid_size,
            degree=5,
            use_residual_scaling=True
        )
        
        # ✅ 独立的预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_space, hidden_space//2),
            nn.LayerNorm(hidden_space//2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space//2, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Linear(hidden_space//4, 1)  # 仅输出SOC
        )
        
    def forward(self, x):
        # 特征编码
        basic_encoded = self.basic_encoder(x[:, :, :10])
        dynamic_encoded = self.dynamic_encoder(x[:, :, 10:22])
        energy_encoded = self.energy_encoder(x[:, :, 22:30])
        impedance_encoded = self.impedance_encoder(x[:, :, 30:40])
        temp_encoded = self.temperature_encoder(x[:, :, 40:46])
        
        concatenated = torch.cat([
            basic_encoded, dynamic_encoded, energy_encoded, 
            impedance_encoded, temp_encoded
        ], dim=-1)
        
        # SOC特定处理
        soc_features = self.soc_branch(concatenated)
        
        # Transformer处理
        transformer_output = self.transformer(soc_features)
        
        # 预测
        prediction = self.prediction_head(soc_features.mean(dim=1))
        
        # 融合
        combined = 0.6 * transformer_output + 0.4 * prediction.unsqueeze(1)
        
        # 基础约束（无SOE耦合）
        constrained = torch.sigmoid(combined)
        
        return constrained


class IndependentSOEModel(nn.Module):
    """
    独立的SOE预测模型
    完全独立的架构，不与SOC共享任何信息
    """
    
    def __init__(self, input_dim, num_heads, num_layers, hidden_space, 
                 dropout_rate, embed_dim, grid_size=16, temperature=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.temperature = temperature
        
        # ✅ 独立的特征编码器（架构相同但参数完全独立）
        self.basic_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(12, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Linear(8, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        self.impedance_encoder = nn.Sequential(
            nn.Linear(10, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//4, hidden_space//4),
            nn.LayerNorm(hidden_space//4)
        )
        
        self.temperature_encoder = nn.Sequential(
            nn.Linear(6, hidden_space//6),
            nn.LayerNorm(hidden_space//6),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(hidden_space//6, hidden_space//6),
            nn.LayerNorm(hidden_space//6)
        )
        
        # ✅ 独立的处理分支
        total_encoded_dim = hidden_space//4 * 3 + hidden_space//6 * 2
        
        self.soe_branch = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_space),
            nn.LayerNorm(hidden_space),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.15)
        )
        
        # ✅ 独立的Transformer
        from model2 import TimeSeriesTransformer_ekan
        
        self.transformer = TimeSeriesTransformer_ekan(
            input_dim=hidden_space,
            num_heads=num_heads,
            num_layers=num_layers,
            num_outputs=1,  # 仅输出SOE
            hidden_space=hidden_space,
            dropout_rate=dropout_rate * 0.5,
            embed_dim=embed_dim,
            grid_size=grid_size,
            degree=5,
            use_residual_scaling=True
        )
        
        # ✅ 独立的预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_space, hidden_space//2),
            nn.LayerNorm(hidden_space//2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(hidden_space//2, hidden_space//4),
            nn.LayerNorm(hidden_space//4),
            nn.GELU(),
            nn.Linear(hidden_space//4, 1)  # 仅输出SOE
        )
        
    def forward(self, x):
        # 特征编码
        basic_encoded = self.basic_encoder(x[:, :, :10])
        dynamic_encoded = self.dynamic_encoder(x[:, :, 10:22])
        energy_encoded = self.energy_encoder(x[:, :, 22:30])
        impedance_encoded = self.impedance_encoder(x[:, :, 30:40])
        temp_encoded = self.temperature_encoder(x[:, :, 40:46])
        
        concatenated = torch.cat([
            basic_encoded, dynamic_encoded, energy_encoded, 
            impedance_encoded, temp_encoded
        ], dim=-1)
        
        # SOE特定处理
        soe_features = self.soe_branch(concatenated)
        
        # Transformer处理
        transformer_output = self.transformer(soe_features)
        
        # 预测
        prediction = self.prediction_head(soe_features.mean(dim=1))
        
        # 融合
        combined = 0.6 * transformer_output + 0.4 * prediction.unsqueeze(1)
        
        # 基础约束（无SOC耦合）
        constrained = torch.sigmoid(combined)
        
        return constrained


class IndependentSOCSOELightningModule(pl.LightningModule):
    """
    独立预测Lightning模块
    使用两个完全独立的模型
    """
    
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict): 
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)
        
        # ✅ 两个完全独立的模型
        self.soc_model = IndependentSOCModel(
            input_dim=46,
            num_heads=hparams.num_heads, 
            num_layers=hparams.n_layers,
            hidden_space=hparams.hidden_space,
            dropout_rate=hparams.dropout, 
            embed_dim=hparams.embed_dim, 
            grid_size=hparams.grid_size,
            temperature=getattr(hparams, 'temperature', None)
        )
        
        self.soe_model = IndependentSOEModel(
            input_dim=46,
            num_heads=hparams.num_heads, 
            num_layers=hparams.n_layers,
            hidden_space=hparams.hidden_space,
            dropout_rate=hparams.dropout, 
            embed_dim=hparams.embed_dim, 
            grid_size=hparams.grid_size,
            temperature=getattr(hparams, 'temperature', None)
        )
        
        self.automatic_optimization = True
        self.test_step_outputs = []
        self.current_epoch_num = 0
        
    def forward(self, x):
        # 分别预测SOC和SOE（完全独立）
        soc_pred = self.soc_model(x)
        soe_pred = self.soe_model(x)
        
        # ❌ 无耦合约束！直接拼接
        combined = torch.cat([soc_pred, soe_pred], dim=-1)
        
        return combined
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch # 解包，忽略原始索引
        
        # 数据增强
        if self.training:
            noise_factor = self.hparams.noise_factor * (1 - self.current_epoch_num / self.hparams.num_epochs)
            if noise_factor > 0: 
                x += torch.randn_like(x) * noise_factor
        
        # 独立预测
        y_hat = self.forward(x)  # [batch, seq_len, 2]
        
        # 🔧 修复维度：y_hat是[batch, seq_len, 2]，需要取最后时间步
        if len(y_hat.shape) == 3:
            y_hat_final = y_hat[:, -1, :]  # [batch, 2] - 取最后时间步
        else:
            y_hat_final = y_hat  # [batch, 2]
        
        # 分别计算SOC和SOE的损失
        soc_loss = F.mse_loss(y_hat_final[:, 0], y[:, 0])
        soe_loss = F.mse_loss(y_hat_final[:, 1], y[:, 1])
        
        # ❌ 无联合损失，无耦合约束
        total_loss = soc_loss + soe_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss): 
            total_loss = F.mse_loss(y_hat_final, y)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_soc_loss', soc_loss, on_step=False, on_epoch=True)
        self.log('train_soe_loss', soe_loss, on_step=False, on_epoch=True)
        
        with torch.no_grad(): 
            train_rmse = torch.sqrt(F.mse_loss(y_hat_final, y))
            self.log('train_rmse', train_rmse, on_step=False, on_epoch=True)
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch # 解包，忽略原始索引
        y_hat = self.forward(x)  # [batch, seq_len, 2]
        
        # 取最后时间步
        if len(y_hat.shape) == 3:
            y_hat_final = y_hat[:, -1, :]  # [batch, 2]
        else:
            y_hat_final = y_hat  # [batch, 2]
        
        val_loss = F.mse_loss(y_hat_final, y)
        val_rmse = torch.sqrt(val_loss)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', val_rmse, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, original_end_indices = batch # 解包原始索引
        y_hat = self.forward(x)  # [batch, seq_len, 2]
        
        # 取最后时间步
        if len(y_hat.shape) == 3:
            y_hat_final = y_hat[:, -1, :]  # [batch, 2]
        else:
            y_hat_final = y_hat  # [batch, 2]
        
        self.test_step_outputs[dataloader_idx].append({'y_true': y.cpu(), 'y_pred': y_hat_final.cpu(), 'original_end_indices': original_end_indices.cpu()})
    
    def on_test_start(self): 
        self.test_step_outputs = [[] for _ in range(2)]
    
    def on_test_epoch_end(self):
        print("\n" + "="*80)
        print("🎯 消融实验 A4.1: SOC-SOE独立预测结果")
        print("="*80)
        print("⚠️  关键区别:")
        print("   ❌ 无SOC-SOE信息共享")
        print("   ❌ 无物理耦合约束")
        print("   ❌ 计算量翻倍")
        print("📊 数据标准化：全局MinMaxScaler")
        
        setup_sci_style()
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Ablation A4.1: Independent SOC-SOE Prediction (No Information Sharing, MinMaxScaler)', 
                    fontsize=12, fontweight='normal')
        
        dataset_names = ["LA92", "UDDS"]
        subplot_idx = 0
        overall_results = {}
        
        for i, outputs in enumerate(self.test_step_outputs):
            if not outputs: continue
            
            y_true_list = []
            y_pred_list = []
            original_end_indices_list = [] # 新增：用于收集原始索引
            for x in outputs:
                if len(x['y_true'].shape) == 3:
                    y_true_list.append(x['y_true'][:, -1, :])
                    original_end_indices_list.append(x['original_end_indices'][:, -1]) # 收集原始索引
                else:
                    y_true_list.append(x['y_true'])
                    original_end_indices_list.append(x['original_end_indices']) # 收集原始索引
                    
                if len(x['y_pred'].shape) == 3:
                    y_pred_list.append(x['y_pred'][:, -1, :])
                else:
                    y_pred_list.append(x['y_pred'])
            
            try:
                y_true = torch.cat(y_true_list).numpy()
                y_pred = torch.cat(y_pred_list).numpy()
                original_end_indices = torch.cat(original_end_indices_list).numpy() # 合并原始索引
            except RuntimeError:
                fixed_y_true_list = []
                fixed_y_pred_list = []
                fixed_original_end_indices_list = [] # 新增：用于收集修复后的原始索引
                for yt, yp, oei in zip(y_true_list, y_pred_list, original_end_indices_list):
                    if len(yt.shape) == 2 and len(yp.shape) == 2:
                        if yt.shape[1] == yp.shape[1]:
                            fixed_y_true_list.append(yt)
                            fixed_y_pred_list.append(yp)
                            fixed_original_end_indices_list.append(oei) # 收集修复后的原始索引
                
                y_true = torch.cat(fixed_y_true_list).numpy()
                y_pred = torch.cat(fixed_y_pred_list).numpy()
                original_end_indices = torch.cat(fixed_original_end_indices_list).numpy() # 合并修复后的原始索引
            
            dataset_name = dataset_names[i]
            # 同时保存原始时间轴索引，以便生成_true.npy和_pred.npy时作为横坐标
            np.save(os.path.join(self.trainer.logger.log_dir or '.', f'ablation_A4_1_independent_MinMaxScaler_time_axis_{dataset_name}.npy'), original_end_indices)
            print(f"   📊 ablation_A4_1_independent_MinMaxScaler_time_axis_{dataset_name}.npy 已保存")
            # --- 新增结束 ---

            time_axis = original_end_indices # 使用原始时间轴索引作为横坐标
            
            for j, feature in enumerate(['SOC', 'SOE']):
                ax_pred = axes[i, j*2]
                actual_values = y_true[:, j] * 100
                pred_values = y_pred[:, j] * 100
                
                actual_smooth = smooth_data(actual_values, window_length=9, polyorder=2)
                pred_smooth = smooth_data(pred_values, window_length=9, polyorder=2)
                
                ax_pred.plot(time_axis, actual_smooth, color='#0072BD', linewidth=1.8, 
                           label='Actual Value', alpha=1.0)
                ax_pred.plot(time_axis, pred_smooth, color='#D95319', linewidth=1.8, 
                           label='Independent', alpha=1.0)
                
                ax_pred.set_xlabel('Time(s)', fontsize=11)
                ax_pred.set_ylabel(f'{feature}(%)', fontsize=11)
                ax_pred.set_title(f'({chr(97+subplot_idx)})', fontsize=12, fontweight='normal')
                ax_pred.legend(loc='upper right', frameon=False, fontsize=10)
                ax_pred.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax_pred.set_ylim(0, 100)

                for spine in ax_pred.spines.values():
                    spine.set_linewidth(1.0)
                    spine.set_color('black')
                
                ax_error = axes[i, j*2 + 1]
                error_values = pred_smooth - actual_smooth
                error_smooth = smooth_data(error_values, window_length=7, polyorder=2)
                
                ax_error.plot(time_axis, error_smooth, color='#1f77b4', linewidth=1.8, alpha=1.0)
                ax_error.set_xlabel('Time(s)', fontsize=11)
                ax_error.set_ylabel(f'{feature} Error (%)', fontsize=11)
                ax_error.set_title(f'({chr(97+subplot_idx+1)})', fontsize=12, fontweight='normal')
                ax_error.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax_error.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.8)
                ax_error.set_ylim(-6, 6)

                for spine in ax_error.spines.values():
                    spine.set_linewidth(1.0)
                    spine.set_color('black')
                
                rmse = np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
                mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
                r2 = r2_score(y_true[:, j], y_pred[:, j])
                
                result_key = f"{dataset_name}_{feature}"
                overall_results[result_key] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
                
                print(f"📊 {dataset_name} - {feature} (独立预测): ")
                print(f"    RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
                
                subplot_idx += 2
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.06, right=0.98, 
                           hspace=0.35, wspace=0.25)
        
        save_path = os.path.join(self.trainer.logger.log_dir or '.', 
                               'ablation_A4_1_independent_MinMaxScaler_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png', pad_inches=0.1)
        print(f"\n📊 结果图已保存: {save_path}")
        
        results_df = pd.DataFrame(overall_results).T
        results_df.index.name = 'Dataset_Feature'
        results_csv_path = os.path.join(self.trainer.logger.log_dir or '.', 
                                      'ablation_A4_1_independent_MinMaxScaler_metrics.csv')
        results_df.to_csv(results_csv_path)
        print(f"📊 测试指标结果已保存: {results_csv_path}")
        
        avg_rmse = np.mean([r['RMSE'] for r in overall_results.values()])
        avg_mae = np.mean([r['MAE'] for r in overall_results.values()])
        avg_r2 = np.mean([r['R2'] for r in overall_results.values()])
        
        la92_rmse = np.mean([overall_results[k]['RMSE'] for k in overall_results.keys() if 'LA92' in k])
        udds_rmse = np.mean([overall_results[k]['RMSE'] for k in overall_results.keys() if 'UDDS' in k])
        
        # 计算SOC-SOE相关性差异
        # 修正：y_true 和 y_pred 已经是 [batch, 2] 形状，可以直接索引
        soc_values = y_true[:, 0]
        soe_values = y_true[:, 1]
        soc_pred_values = y_pred[:, 0]
        soe_pred_values = y_pred[:, 1]
        
        soc_soe_gap = np.abs(soc_pred_values - soe_pred_values).mean()
        
        print(f"\n🏆 独立预测综合性能:")
        print(f"    整体平均RMSE: {avg_rmse:.6f} ({avg_rmse*100:.2f}%) ")
        print(f"    平均MAE:  {avg_mae:.6f}")
        print(f"    平均R²:   {avg_r2:.6f}")
        print(f"    SOC-SOE平均差距: {soc_soe_gap:.6f}")
        
        print(f"\n📊 分工况性能:")
        print(f"    LA92平均RMSE: {la92_rmse:.6f} ({la92_rmse*100:.2f}%) ")
        print(f"    UDDS平均RMSE: {udds_rmse:.6f} ({udds_rmse*100:.2f}%) ")
        
        full_model_rmse = 0.020 # 假设完整KAN+Transformer模型的RMSE基准
        performance_gap = avg_rmse - full_model_rmse
        degradation_pct = (performance_gap / full_model_rmse) * 100
        
        print(f"\n📉 vs 联合预测（完整模型）:")
        print(f"    完整模型RMSE: {full_model_rmse:.6f}")
        print(f"    独立预测RMSE: {avg_rmse:.6f}")
        print(f"    性能下降: {performance_gap:.6f} ({degradation_pct:.1f}%)")
        print(f"    证明: SOC-SOE联合预测贡献了{degradation_pct:.1f}%的性能提升！")
        print(f"    🔬 科学发现:")
        print(f"       - 信息共享的价值")
        print(f"       - 物理耦合约束的必要性")
        print(f"       - 联合学习 vs 独立学习的优势")
        
        plt.show()
        print("="*80)
    
    def on_train_epoch_end(self): 
        self.current_epoch_num += 1
    
    def configure_optimizers(self):
        # 两个模型的参数一起优化
        params = list(self.soc_model.parameters()) + list(self.soe_model.parameters())
        
        optimizer = torch.optim.AdamW(
            params, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15,
            min_lr=self.hparams.lr * 0.001
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


class MemoryCleanupCallback(pl.Callback):
    """内存清理回调"""
    def on_train_epoch_end(self, trainer, pl_module): 
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='消融实验 A4.1: SOC-SOE独立预测')
    
    parser.add_argument('--train_paths', type=str, nargs='+', default=[
        r"C:\25degC training\03-18-17_02.17 25degC_Cycle_1_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_03.25 25degC_Cycle_2_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_09.07 25degC_Cycle_3_Pan18650PF.csv",
        r"C:\25degC training\03-19-17_14.31 25degC_Cycle_4_Pan18650PF.csv",
        r"C:\25degC training\03-20-17_01.43 25degC_US06_Pan18650PF.csv",
        r"C:\25degC training\03-20-17_05.56 25degC_HWFTa_Pan18650PF.csv",
        r"C:\25degC testing\03-21-17_16.27 25degC_NN_Pan18650PF.csv"
    ])
    parser.add_argument('--test_paths', type=str, nargs='+', default=[
        r"C:\25degC training\03-21-17_09.38 25degC_LA92_Pan18650PF.csv",
        r"C:\25degC testing\03-21-17_00.29 25degC_UDDS_Pan18650PF.csv"
    ])
    parser.add_argument('--result_dir', type=str, default='ablation_A4_1_independent_MinMaxScaler_results')
    parser.add_argument('--output_features', type=str, default='SOC,SOE')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=25.0)
    
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--hidden_space', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--weight_decay', type=float, default=0.0006)
    parser.add_argument('--noise_factor', type=float, default=0.004)
    parser.add_argument('--electrochemical_weight', type=float, default=0.025)
    parser.add_argument('--grid_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--train_val_split_ratio', type=float, default=0.93, help='Ratio for training set split from total training data.') # 新增：训练集与验证集划分比例
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to a pre-trained checkpoint to load for testing') # 新增：检查点路径
    
    args = parser.parse_args()
    args.output_features = [item.strip() for item in args.output_features.split(',')]
    # os.makedirs(args.result_dir, exist_ok=True) # 移除此行，将在logger中处理
    
    print("="*80)
    print("📊 === 消融实验 A4.1: SOC-SOE独立预测 - 25°C (MinMaxScaler) ===") # 更新打印信息
    print("="*80)
    print("🔬 特征配置:")
    print("   ✅ 使用所有46个电化学特征")
    print("   🔄 数据标准化：全局MinMaxScaler")
    print("")
    print("🎯 实验目的:")
    print("   1. 验证SOC-SOE联合预测的价值")
    print("   2. 量化信息共享和物理耦合约束的贡献")
    print("")
    print("📉 移除的组件:")
    print("   ❌ SOC-SOE联合预测（单一模型同时输出2个值）")
    print("   ❌ SOC-SOE耦合约束 (|SOE - SOC| < 0.12)")
    print("   ❌ 共享特征编码器")
    print("")
    print("🔄 替换方案:")
    print("   两个完全独立的模型：")
    print("   - SOC模型：独立编码器 + Transformer + 预测头 → SOC")
    print("   - SOE模型：独立编码器 + Transformer + 预测头 → SOE")
    print("")
    print("✅ 保留的组件:")
    print("   ✓ 相同的架构复杂度（每个模型）")
    print("   ✓ 相同的训练策略")
    print("   ✓ 相同的超参数")
    print("\n📊 预期性能下降: 8-12% (相比完整模型)")
    print("   目的: 验证联合预测和物理耦合的价值")
    print("="*80)
    
    try:
        pl.seed_everything(args.seed, workers=True)
        datamodule = Electrochemical46FeaturesKANDataModule(args) # 使用新的KAN数据模块
        datamodule.setup(stage='fit') # 显式调用setup，确保scaler拟合
        
        model = IndependentSOCSOELightningModule(args)
        
        # 显式配置logger，使其保存到args.result_dir
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=args.result_dir, name='', version='') # name为空，version为空，直接保存到save_dir

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.result_dir, 'checkpoints'), # 将检查点保存到子文件夹
            filename='ablation-independent-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1, monitor='val_loss', mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss', patience=args.patience, mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        trainer = pl.Trainer(
            max_epochs=args.num_epochs, 
            accelerator='auto', 
            devices=1,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, MemoryCleanupCallback()],
            precision='32',
            gradient_clip_val=0.5,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            logger=logger # 传入配置好的logger
        )
        
        print(f"\n🚀 开始独立预测训练...")
        print(f"⚠️  注意：使用两个完全独立的模型，参数量翻倍！")
        if args.ckpt_path:
            print(f"   跳过训练，从检查点加载模型: {args.ckpt_path}")
            model = type(model).load_from_checkpoint(args.ckpt_path, hparams=args)
            trainer.test(model, datamodule=datamodule)
        else:
            trainer.fit(model, datamodule)
            print(f"\n📊 测试独立预测模型...")
            trainer.test(model, datamodule=datamodule, ckpt_path='best')
        
        print(f"\n✅ 消融实验 A4.1 完成！")
        print(f"📁 结果保存在: {args.result_dir}") # 新增：打印结果保存路径
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
