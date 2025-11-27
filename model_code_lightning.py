
import argparse
import gc
import math
import os
import platform
import time
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         ModelCheckpoint)
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# 从您的项目中导入模型定义
from model2 import TimeSeriesTransformer_ekan, TimeSeriesTransformer
from utils import *

# --- Matplotlib 中文显示设置 (与原代码相同) ---
def setup_chinese_font():
    """确保每次绘图前都设置好中文字体"""
    try:
        # 优先寻找并设置中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
        font_found = False
        for font_name in font_list:
            if fm.findfont(fm.FontProperties(family=font_name)):
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                print(f"使用中文字体: {font_name}")
                break
        if not font_found:
            print("警告：未找到推荐的中文字体，图形中的中文可能无法显示。")

        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    except Exception as e:
        print(f"设置中文字体时出错: {e}")

setup_chinese_font()


# --- PyTorch Lightning 数据模块 ---
class BatteryDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

    def setup(self, stage=None):
        # --- REFACTORED FOR SAFE WINDOWING ---

        # 1. 为缩放器加载并合并所有训练数据
        print("--- 正在加载并合并训练数据以拟合缩放器 ---")
        full_train_data_for_scaler = self._load_data(self.hparams.train_paths, "训练集")
        full_train_data_for_scaler = self._preprocess_data(full_train_data_for_scaler)

        # 2. 独立加载训练和测试数据集列表
        train_dfs = self._load_data_as_list(self.hparams.train_paths, "训练集")
        self.raw_test_dfs = self._load_data_as_list(self.hparams.test_paths, "测试集")

        # 3. 独立预处理每个数据集
        processed_train_dfs = [self._preprocess_data(df.copy()) for df in train_dfs]
        processed_test_dfs = [self._preprocess_data(df.copy()) for df in self.raw_test_dfs]
        
        # 4. 在合并后的数据上拟合缩放器以获得全局范围
        print("--- 正在拟合全局缩放器 ---")
        self.scaler.fit(full_train_data_for_scaler[self.hparams.input_features])
        self.target_scaler.fit(full_train_data_for_scaler[self.hparams.output_features])
        
        # 清理大型合并数据以释放内存
        del full_train_data_for_scaler
        gc.collect()

        # 5. 为每个独立的训练数据集创建无污染的滑动窗口
        all_X_train, all_y_train = [], []
        print("--- 正在为每个训练数据集独立创建滑动窗口 ---")
        for i, df in enumerate(processed_train_dfs):
            path = self.hparams.train_paths[i]
            # 使用全局缩放器转换数据
            features_scaled = self.scaler.transform(df[self.hparams.input_features])
            target_scaled = self.target_scaler.transform(df[self.hparams.output_features])
            
            X, y = self._create_windows(features_scaled, target_scaled, self.hparams.window_size)
            
            if X.size > 0:
                all_X_train.append(X)
                all_y_train.append(y)
            else:
                print(f"警告：处理文件 {path} 后未创建任何窗口数据。")

        # 将所有窗口数据合并为一个大的训练集
        X_train = np.concatenate(all_X_train, axis=0)
        y_train = np.concatenate(all_y_train, axis=0)
        print(f"从所有训练文件中总共创建了 {len(X_train)} 个无污染的训练窗口。")

        del all_X_train, all_y_train
        gc.collect()
        
        # 6. 为每个独立的测试数据集创建无污染的滑动窗口
        self.test_datasets = []
        for i, df in enumerate(processed_test_dfs):
            features_scaled = self.scaler.transform(df[self.hparams.input_features])
            target_scaled = self.target_scaler.transform(df[self.hparams.output_features])
            X_test, y_test = self._create_windows(features_scaled, target_scaled, self.hparams.window_size)
            
            if X_test.size > 0:
                X_test_tensor = torch.from_numpy(X_test).type(torch.float32)
                y_test_tensor = torch.from_numpy(y_test).type(torch.float32)
                self.test_datasets.append(TensorDataset(X_test_tensor, y_test_tensor))
            else:
                test_set_name = Path(self.hparams.test_paths[i]).stem
                print(f"警告：创建窗口后，测试集 '{test_set_name}' 为空，将被跳过。")

        # 7. 转换为Tensor并创建最终的训练/验证数据集
        X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
        y_train_tensor = torch.from_numpy(y_train).type(torch.float32)

        full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.hparams.seed)
        )

        print(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")
        for i, ds in enumerate(self.test_datasets):
            test_set_name = Path(self.hparams.test_paths[i]).stem
            print(f"测试集 '{test_set_name}' 大小: {len(ds)}")

    def _load_data(self, paths, name):
        df_list = []
        print(f"--- 正在加载 {name} 数据 ---")
        for path in paths:
            if os.path.exists(path):
                df_list.append(pd.read_csv(path))
            else:
                print(f"警告: 文件不存在: {path}")
        if not df_list:
            raise FileNotFoundError(f"未成功加载任何{name}数据，请检查路径。")
        return pd.concat(df_list, ignore_index=True)

    # --- NEW: Helper to load dataframes into a list ---
    def _load_data_as_list(self, paths, name):
        df_list = []
        print(f"--- 正在加载 {name} 数据 ---")
        for path in paths:
            if os.path.exists(path):
                df_list.append(pd.read_csv(path))
            else:
                print(f"警告: 文件不存在: {path}")
        if not df_list:
            raise FileNotFoundError(f"未成功加载任何{name}数据，请检查路径。")
        return df_list

    def _preprocess_data(self, df):
        if 'Current' in df.columns:
            df = df[df['Current'] != 0].copy()
        
        features_to_check = self.hparams.input_features + self.hparams.output_features
        df.dropna(subset=features_to_check, inplace=True)
        
        # 填充其他列的NaN
        if df.isnull().values.any():
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.fillna(0, inplace=True)
            
        return df.reset_index(drop=True)

    def _create_windows(self, features, targets, window_size):
        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i + window_size])
            y.append(targets[i + window_size])
        return np.array(X), np.array(y)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True if self.hparams.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True if self.hparams.num_workers > 0 else False)

    # --- MODIFIED: Return a list of dataloaders for testing ---
    def test_dataloader(self):
        return [DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True if self.hparams.num_workers > 0 else False) for ds in self.test_datasets]


# --- PyTorch Lightning 模型模块 ---
class KANTransformerLightning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        
        # 用于存储测试结果
        self.test_step_outputs = []

    def _build_model(self):
        if self.hparams.model_name == 'Transformer-ekan':
            return TimeSeriesTransformer_ekan(
                input_dim=len(self.hparams.input_features),
                num_heads=self.hparams.num_heads,
                num_layers=self.hparams.n_layers,
                num_outputs=len(self.hparams.output_features),
                hidden_space=self.hparams.hidden_space,
                dropout_rate=self.hparams.dropout,
                embed_dim=self.hparams.embed_dim,
                grid_size=self.hparams.get('grid_size', 16) # 使用.get()方法安全地访问参数
            )
        else: # Fallback to standard transformer
            return TimeSeriesTransformer(
                input_dim=len(self.hparams.input_features),
                num_heads=self.hparams.num_heads,
                num_layers=self.hparams.n_layers,
                num_outputs=len(self.hparams.output_features),
                hidden_space=self.hparams.hidden_space,
                dropout_rate=self.hparams.dropout
            )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat.detach()

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # 数据增强：在训练时添加轻微噪声
        if hasattr(self.hparams, 'noise_factor') and self.hparams.noise_factor > 0:
            noise = torch.randn_like(x) * self.hparams.noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x
            
        y_hat = self(x_noisy)
        loss = self.criterion(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # 新增：计算并记录 train_rmse
        train_rmse = torch.sqrt(loss)
        self.log('train_rmse', train_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        # 计算 R² score
        r2 = r2_score(y.cpu().numpy(), y_hat.cpu().numpy())
        self.log('val_r2', r2, on_epoch=True, prog_bar=True, logger=True)
        
    # --- MODIFIED: Accept dataloader_idx for multi-dataset testing ---
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, y, y_hat = self._shared_step(batch, batch_idx)
        # Log loss for each test dataloader separately
        self.log(f'test_loss_loader_{dataloader_idx}', loss, on_step=False, on_epoch=True)
        self.test_step_outputs[dataloader_idx].append({'y_true': y.cpu(), 'y_pred': y_hat.cpu()})
        
    # --- NEW: Prepare for multi-dataset testing ---
    def on_test_start(self):
        """在测试开始前，根据测试数据加载器的数量初始化输出列表"""
        # Ensure test_dataloaders is a list
        if hasattr(self.trainer, 'test_dataloaders') and self.trainer.test_dataloaders:
            num_test_dataloaders = len(self.trainer.test_dataloaders)
        else: # Fallback for older lightning versions or different contexts
            num_test_dataloaders = 1
        self.test_step_outputs = [[] for _ in range(num_test_dataloaders)]

    # --- NEW: Helper to robustly find the time column ---
    def _get_time_column(self, df):
        """Tries to find the time column from a list of common names."""
        possible_names = ['Time (s)', 'time(s)', 'Time(s)', 'time', 'Time']
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # --- MODIFIED: Complete overhaul for per-dataset evaluation and plotting ---
    def on_test_epoch_end(self):
        dm = self.trainer.datamodule
        results_per_dataset = []
        test_set_names = [Path(p).stem for p in self.hparams.test_paths]

        # --- NEW: Also evaluate on the training set for the final report ---
        print("\n--- 正在评估最终训练集性能 ---")
        train_loader = dm.train_dataloader()
        y_true_train_list, y_pred_train_list = [], []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="评估训练集"):
                x, y = batch
                x = x.to(self.device)
                y_hat = self.model(x)
                y_true_train_list.append(y.cpu())
                y_pred_train_list.append(y_hat.cpu())
        
        y_true_train = torch.cat(y_true_train_list).numpy()
        y_pred_train = torch.cat(y_pred_train_list).numpy()
        y_true_train_denorm = dm.target_scaler.inverse_transform(y_true_train)
        y_pred_train_denorm = dm.target_scaler.inverse_transform(y_pred_train)
        
        train_metrics = {}
        for j, feature in enumerate(self.hparams.output_features):
            rmse = np.sqrt(mean_squared_error(y_true_train_denorm[:, j], y_pred_train_denorm[:, j]))
            r2 = r2_score(y_true_train_denorm[:, j], y_pred_train_denorm[:, j])
            mae = mean_absolute_error(y_true_train_denorm[:, j], y_pred_train_denorm[:, j])
            train_metrics[feature] = {'rmse': rmse, 'r2': r2, 'mae': mae}


        # --- Existing test set evaluation ---
        all_y_true_test_denorm, all_y_pred_test_denorm = [], []
        for i, outputs in enumerate(self.test_step_outputs):
            if not outputs:
                print(f"警告: 测试集 {test_set_names[i]} 没有输出，跳过评估。")
                continue
            
            y_true = torch.cat([x['y_true'] for x in outputs]).numpy()
            y_pred = torch.cat([x['y_pred'] for x in outputs]).numpy()
            
            # 反归一化
            y_true_denorm = dm.target_scaler.inverse_transform(y_true)
            y_pred_denorm = dm.target_scaler.inverse_transform(y_pred)
            
            # --- NEW: Collect all test results for combined metrics ---
            all_y_true_test_denorm.append(y_true_denorm)
            all_y_pred_test_denorm.append(y_pred_denorm)

            # 计算指标
            metrics = {}
            print(f"\n--- 最终测试评估结果 ({test_set_names[i]}) ---")
            for j, feature in enumerate(self.hparams.output_features):
                rmse = np.sqrt(mean_squared_error(y_true_denorm[:, j], y_pred_denorm[:, j]))
                r2 = r2_score(y_true_denorm[:, j], y_pred_denorm[:, j])
                mae = mean_absolute_error(y_true_denorm[:, j], y_pred_denorm[:, j])
                metrics[feature] = {'rmse': rmse, 'r2': r2, 'mae': mae}
                print(f"{feature}: RMSE={rmse:.6f}, R2={r2:.6f}, MAE={mae:.6f}")

            # 获取绘图所需的时间轴
            processed_df = dm._preprocess_data(dm.raw_test_dfs[i].copy())
            time_col_name = self._get_time_column(processed_df)
            
            time_axis = None
            if time_col_name:
                time_axis = processed_df[time_col_name].iloc[self.hparams.window_size:].values
                if len(time_axis) != len(y_true_denorm):
                    print(f"警告：时间轴 ({len(time_axis)}) 和数据 ({len(y_true_denorm)}) 长度不匹配，绘图将使用样本索引。")
                    time_axis = np.arange(len(y_true_denorm))
            else:
                print(f"警告：在测试集 {test_set_names[i]} 中未找到时间列，绘图将使用样本索引。")
                time_axis = np.arange(len(y_true_denorm))

            results_per_dataset.append({
                'name': test_set_names[i],
                'y_true': y_true_denorm,
                'y_pred': y_pred_denorm,
                'metrics': metrics,
                'time_axis': time_axis
            })

        self.test_step_outputs.clear() # free memory

        # 绘制结果图并写入总结
        if results_per_dataset:
            # --- NEW: Calculate combined test metrics ---
            final_y_true_test = np.concatenate(all_y_true_test_denorm, axis=0)
            final_y_pred_test = np.concatenate(all_y_pred_test_denorm, axis=0)
            combined_test_metrics = {}
            for j, feature in enumerate(self.hparams.output_features):
                rmse = np.sqrt(mean_squared_error(final_y_true_test[:, j], final_y_pred_test[:, j]))
                r2 = r2_score(final_y_true_test[:, j], final_y_pred_test[:, j])
                mae = mean_absolute_error(final_y_true_test[:, j], final_y_pred_test[:, j])
                combined_test_metrics[feature] = {'rmse': rmse, 'r2': r2, 'mae': mae}

            self.plot_publication_style_results(results_per_dataset)
            self.write_evaluation_summary(results_per_dataset, train_metrics, combined_test_metrics)

    # --- NEW: Publication-style plotting function ---
    def plot_publication_style_results(self, results_per_dataset):
        print("正在生成发布风格的结果图...")
        save_path = os.path.join(self.hparams.result_dir, 'publication_style_results.png')
        
        num_sets = len(results_per_dataset)
        if num_sets == 0: return
            
        fig, axes = plt.subplots(num_sets, 4, figsize=(24, 6 * num_sets), squeeze=False)
        fig.suptitle('Actual and estimate values with different drive cycles under 20° C', fontsize=18)

        subplot_labels = 'abcdefgh'
        label_idx = 0

        for i, result in enumerate(results_per_dataset):
            y_true, y_pred, time_axis = result['y_true'], result['y_pred'], result['time_axis']
            
            # --- FIX: Convert SOC/SOE values from [0, 1] to [0, 100] for plotting ---
            y_true_percent = y_true * 100
            y_pred_percent = y_pred * 100

            y_axis_name = "LA92" if 'la92' in result['name'].lower() else "UDDS" if 'udds' in result['name'].lower() else f"Test Set {i+1}"

            # Plot SOC
            ax_soc = axes[i, 0]
            ax_soc.plot(time_axis, y_true_percent[:, 0], 'b-', label='Actual Value', linewidth=2)
            ax_soc.plot(time_axis, y_pred_percent[:, 0], 'r--', label='Estimated Value', linewidth=2)
            ax_soc.set(ylabel=f'{y_axis_name}\nSOC (%)', xlabel='Time (s)', ylim=(0, 105), title=f'({subplot_labels[label_idx]})')
            ax_soc.legend()
            label_idx += 1

            # Plot SOE
            ax_soe = axes[i, 1]
            ax_soe.plot(time_axis, y_true_percent[:, 1], 'b-', label='Actual Value', linewidth=2)
            ax_soe.plot(time_axis, y_pred_percent[:, 1], 'r--', label='Estimated Value', linewidth=2)
            ax_soe.set(ylabel='SOE (%)', xlabel='Time (s)', ylim=(0, 105), title=f'({subplot_labels[label_idx]})')
            ax_soe.legend()
            label_idx += 1

            # Plot SOC Error
            ax_soc_err = axes[i, 2]
            soc_error = y_pred_percent[:, 0] - y_true_percent[:, 0]
            ax_soc_err.plot(time_axis, soc_error, color='#3685b5', linewidth=1.5)
            ax_soc_err.set(ylabel='SOC Error (%)', xlabel='Time (s)', ylim=(-20, 20), title=f'({subplot_labels[label_idx]})')
            label_idx += 1
            
            # Plot SOE Error
            ax_soe_err = axes[i, 3]
            soe_error = y_pred_percent[:, 1] - y_true_percent[:, 1]
            ax_soe_err.plot(time_axis, soe_error, color='#3685b5', linewidth=1.5)
            ax_soe_err.set(ylabel='SOE Error (%)', xlabel='Time (s)', ylim=(-20, 20), title=f'({subplot_labels[label_idx]})')
            label_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存图表: {save_path}")

    # --- MODIFIED: Function to write summary file with more details ---
    def write_evaluation_summary(self, results_per_dataset, train_metrics, combined_test_metrics):
        summary_path = os.path.join(self.hparams.result_dir, 'evaluation_summary.txt')
        print(f"正在写入评估总结: {summary_path}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"模型: {self.hparams.model_name}\n")
            f.write(f"输入特征: {self.hparams.input_features}\n")
            f.write(f"输出特征: {self.hparams.output_features}\n")
            f.write(f"窗口大小: {self.hparams.window_size}\n\n")
            
            f.write("训练数据集:\n")
            for path in self.hparams.train_paths:
                f.write(f"  - {path}\n")
            f.write("\n测试数据集:\n")
            for path in self.hparams.test_paths:
                f.write(f"  - {path}\n")

            f.write("\n训练配置:\n")
            f.write(f"  - 设备: {'gpu' if 'cuda' in str(self.device) else 'cpu'}\n")
            f.write(f"  - 混合精度训练: {'启用' if self.hparams.use_amp else '禁用'}\n")
            f.write(f"  - 训练集分割: 80% 训练, 20% 验证\n")
            if 'cuda' in str(self.device):
                 f.write(f"  - GPU型号: {torch.cuda.get_device_name(0)}\n")

            f.write("\n超参数:\n")
            f.write(f"  - 注意力头数: {self.hparams.num_heads}\n")
            f.write(f"  - 层数: {self.hparams.n_layers}\n")
            f.write(f"  - Dropout率: {self.hparams.dropout:.4f}\n")
            f.write(f"  - 隐藏空间维度: {self.hparams.hidden_space}\n")
            f.write(f"  - 嵌入维度: {self.hparams.embed_dim}\n")
            f.write(f"  - 批量大小: {self.hparams.batch_size}\n")
            f.write(f"  - 学习率: {self.hparams.lr}\n")
            
            f.write("\n评估结果:\n")
            
            f.write("\n训练集:\n")
            for feature, metrics in train_metrics.items():
                f.write(f"  - {feature}: RMSE = {metrics['rmse']:.6f}, R² = {metrics['r2']:.6f}, MAE = {metrics['mae']:.6f}\n")

            f.write("\n测试集 (合并所有工况):\n")
            for feature, metrics in combined_test_metrics.items():
                f.write(f"  - {feature}: RMSE = {metrics['rmse']:.6f}, R² = {metrics['r2']:.6f}, MAE = {metrics['mae']:.6f}\n")

            f.write("\n测试集 (分工况):\n")
            for result in results_per_dataset:
                f.write(f"\n-- 工况: {Path(self.hparams.test_paths[results_per_dataset.index(result)]).name} --\n")
                for feature, metrics in result['metrics'].items():
                    f.write(f"  - {feature}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}, MAE={metrics['mae']:.6f}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.get("weight_decay", 0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.3,  # 更激进的学习率衰减
            patience=3,  # 更早开始衰减学习率
            min_lr=1e-6  # 设置最小学习率
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def main(args):
    pl.seed_everything(args.seed, workers=True)
    
    # 将 Namespace 转换为字典，以便传递给 Lightning 模块
    hparams = args
    
    datamodule = BatteryDataModule(hparams)
    model = KANTransformerLightning(hparams)
    
    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.result_dir,
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    
    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(args.result_dir, 'logs')),
        precision="16-mixed" if args.use_amp else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=10,
    )
    
    print("--- 开始训练 ---")
    trainer.fit(model, datamodule, ckpt_path=args.resume_from_checkpoint)
    
    print("--- 开始测试 ---")
    trainer.test(model, datamodule=datamodule, ckpt_path='best')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用PyTorch Lightning训练KAN-Transformer模型')
    
    # Data - 恢复为最佳实验的配置
    parser.add_argument('--train_paths', type=str, nargs='+', default=[
        r"C:\0degC training\06-01-17_15.36 0degC_Cycle_3_Pan18650PF.csv",
        r"C:\0degC training\06-01-17_22.03 0degC_Cycle_4_Pan18650PF.csv",
        r"C:\0degC training\06-02-17_04.58 0degC_US06_Pan18650PF.csv",
        r"C:\0degC training\06-02-17_10.43 0degC_HWFET_Pan18650PF.csv",
        r"C:\0degC training\05-30-17_12.56 0degC_Cycle_1_Pan18650PF.csv",
        r"C:\0degC training\05-30-17_20.16 0degC_Cycle_2_Pan18650PF.csv",
        r"C:\0degC training\06-01-17_10.36 0degC_NN_Pan18650PF.csv"
    ])
    parser.add_argument('--test_paths', type=str, nargs='+', default=[
        r"C:\0degC testing\06-01-17_10.36 0degC_LA92_NN_Pan18650PF.csv",
        r"C:\0degC testing\06-02-17_17.14 0degC_UDDS_Pan18650PF.csv"
    ])
    parser.add_argument('--result_dir', type=str, default=r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\0degC")
    parser.add_argument('--input_features', type=str, default='Voltage,Current')
    parser.add_argument('--output_features', type=str, default='SOC,SOE')
    parser.add_argument('--window_size', type=int, default=20)

    # Model - 恢复为最佳实验的配置
    parser.add_argument('--model_name', type=str, default='Transformer-ekan')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_space', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=57)
    parser.add_argument('--dropout', type=float, default=0.25)  # 增加dropout以减少过拟合
    parser.add_argument('--grid_size', type=int, default=16) # 恢复为与最佳模型一致的16
    
    # Training - 针对过拟合优化的配置
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=12) # 减少早停耐心值以防止过拟合
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--use_amp', action='store_true') # 最佳实验禁用，所以不设置默认值
    parser.add_argument('--num_workers', type=int, default=0) # 为兼容Windows设为0
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4) # 添加权重衰减以减少过拟合
    parser.add_argument('--noise_factor', type=float, default=0.01) # 添加数据增强噪声参数
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='用于恢复训练的检查点路径')
 
    args = parser.parse_args()
    
    # 将逗号分隔的字符串特征转换为列表
    args.input_features = [item.strip() for item in args.input_features.split(',')]
    args.output_features = [item.strip() for item in args.output_features.split(',')]

    # 确保hidden_space可以被num_heads整除
    if args.hidden_space % args.num_heads != 0:
        args.hidden_space = (args.hidden_space // args.num_heads) * args.num_heads
        print(f"调整hidden_space为{args.hidden_space}以确保能被num_heads整除")

    main(args)
