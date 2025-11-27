import argparse
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_simple_features(csv_paths, window_size=32, overlap_ratio=0.5, is_test_set=False,
                                        scaler_v=None, scaler_c=None):
    """
    加载数据并提取2个基础特征：电压、电流
    
    返回:
        features: [num_samples, window_size, 2] (Voltage, Current)
        targets: [num_samples, 2] (SOC, SOE)
        original_end_indices: [num_samples] (原始序列中窗口结束的时间步索引)
    """
    all_features = []
    all_targets = []
    all_original_end_indices = []

    for csv_path in csv_paths:
        # print(f"📂 加载数据: {os.path.basename(csv_path)}") # 可以在这里加一个更详细的日志
        df = pd.read_csv(csv_path)
        # print(f"   📊 原始CSV文件 {os.path.basename(csv_path)} 的行数: {len(df)}")

        # 提取2个基础特征
        voltage = df['Voltage(V)'].values if 'Voltage(V)' in df.columns else df['Voltage'].values
        current = df['Current(A)'].values if 'Current(A)' in df.columns else df['Current'].values

        # === 全局标准化修改点1：不再在此处 fit_transform ===
        # 确保传入了 scaler
        if scaler_v is None or scaler_c is None:
            raise ValueError("必须提供预训练的 StandardScaler (scaler_v, scaler_c) 进行数据转换。")

        # 只进行 transform
        voltage_norm = scaler_v.transform(voltage.reshape(-1, 1)).flatten()
        current_norm = scaler_c.transform(current.reshape(-1, 1)).flatten()
        # === 全局标准化修改点1 结束 ===
        
        # 滑动窗口
        # 根据是否为测试集设置步长
        if is_test_set:
            step_size = 1 # 测试集使用步长为1进行滚动预测
            # print(f"   ⚙️ 测试集模式: 滑动窗口步长设置为 1")
        else:
            step_size = int(window_size * (1 - overlap_ratio))
            if step_size < 1: step_size = 1 # 确保步长至少为1
            # print(f"   ⚙️ 训练/验证集模式: 滑动窗口步长设置为 {step_size}")
        
        # 确保循环至少运行一次，如果数据足够创建一个窗口的话
        max_i = len(df) - window_size
        if max_i < 0: # 如果数据长度不足一个窗口
            # print(f"   ⚠️ 数据文件 {os.path.basename(csv_path)} 长度不足一个窗口 ({len(df)} < {window_size})，跳过.")
            continue # 跳过当前文件
        elif max_i == 0 and window_size == len(df): # 刚好一个窗口
            # 只有一个样本，不需要循环，直接处理
            pass
        # else: # 移除了这个多余的判断
        #     print(f"   ⚠️ 数据文件 {os.path.basename(csv_path)} 长度不足一个窗口 ({len(df)} < {window_size})，跳过.")
        #     continue
        
        for i in range(0, max_i + 1, step_size): # 确保包含最后一个可能的窗口
            # 特征窗口
            v_window = voltage_norm[i:i+window_size]
            c_window = current_norm[i:i+window_size]
            
            feature_window = np.stack([v_window, c_window], axis=-1)  # [window_size, 2]
            all_features.append(feature_window)
            
            # 目标（窗口最后一个时间步）
            target = np.array([df['SOC'].values[i+window_size-1], df['SOE'].values[i+window_size-1]]) # 直接从df取soc/soe
            all_targets.append(target)
            all_original_end_indices.append(i + window_size - 1) # 记录原始序列中窗口结束的时间步索引
    
    features = np.array(all_features)
    targets = np.array(all_targets)
    original_end_indices = np.array(all_original_end_indices)
    
    # print(f"✅ 数据加载完成: 特征维度={features.shape}, 目标维度={targets.shape}, 原始结束索引维度={original_end_indices.shape}")
    # print(f"   特征数量: 2 (Voltage, Current)")
    
    return features, targets, original_end_indices


class SimpleFeatureDataModule(pl.LightningDataModule):
    """数据模块 - 2特征版"""
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.scaler_v = None # 初始化 scaler_v
        self.scaler_c = None # 初始化 scaler_c
        
    def setup(self, stage=None):
        print(f"🧪 开始数据加载 ({self.hparams.temperature}°C, 2特征版)")
        print(f"   特征: Voltage + Current")

        # === 全局标准化修改点2：第一阶段 - 收集训练数据并拟合标准化器 ===
        print("📊 阶段1: 收集所有训练数据并拟合标准化器...")
        all_train_raw_voltage = []
        all_train_raw_current = []
        
        for csv_path in self.hparams.train_paths:
            print(f"   读取原始训练数据: {os.path.basename(csv_path)}")
            df = pd.read_csv(csv_path)
            voltage = df['Voltage(V)'].values if 'Voltage(V)' in df.columns else df['Voltage'].values
            current = df['Current(A)'].values if 'Current(A)' in df.columns else df['Current'].values
            all_train_raw_voltage.extend(voltage)
            all_train_raw_current.extend(current)
            
        # 拟合 StandardScaler
        self.scaler_v = StandardScaler()
        self.scaler_c = StandardScaler()
        
        self.scaler_v.fit(np.array(all_train_raw_voltage).reshape(-1, 1))
        self.scaler_c.fit(np.array(all_train_raw_current).reshape(-1, 1))
        
        print(f"✅ StandardScaler 已在所有训练集 Voltage (mean={self.scaler_v.mean_[0]:.4f}, std={self.scaler_v.scale_[0]:.4f}) 和 Current (mean={self.scaler_c.mean_[0]:.4f}, std={self.scaler_c.scale_[0]:.4f}) 上拟合完成。")
        # === 全局标准化修改点2 结束 ===
        
        # === 全局标准化修改点3：第二阶段 - 使用 fit 好的 scaler 加载并转换训练数据 ===
        print("📊 阶段2: 使用拟合的标准化器转换训练集和验证集数据...")
        train_features, train_targets, train_original_end_indices = load_and_preprocess_simple_features(
            self.hparams.train_paths,
            window_size=self.hparams.window_size,
            overlap_ratio=self.hparams.overlap_ratio,
            is_test_set=False, # 训练集步长仍按 overlap_ratio 计算
            scaler_v=self.scaler_v, # 传入 fit 好的 scaler
            scaler_c=self.scaler_c  # 传入 fit 好的 scaler
        )
        
        # 数据转换
        X_train_tensor = torch.from_numpy(train_features).float()
        y_train_tensor = torch.from_numpy(train_targets).float()
        
        # 数据划分
        dataset_size = len(X_train_tensor)
        train_size = int(0.93 * dataset_size)  # 93% 训练
        val_size = dataset_size - train_size   # 7% 验证
        print(f"📊 数据划分 (93%/7%): 训练={train_size:,}, 验证={val_size:,}")
        
        full_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, torch.from_numpy(train_original_end_indices).long())
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        # === 全局标准化修改点3 结束 ===
        
        # === 全局标准化修改点4：第二阶段 - 使用 fit 好的 scaler 加载并转换测试数据 ===
        print("📊 阶段3: 使用拟合的标准化器转换测试集数据...")
        self.test_datasets = []
        for test_path in self.hparams.test_paths:
            test_features, test_targets, test_original_end_indices = load_and_preprocess_simple_features(
                [test_path],
                window_size=self.hparams.window_size,
                overlap_ratio=self.hparams.overlap_ratio, 
                is_test_set=True, # 测试集步长为1
                scaler_v=self.scaler_v, # 传入 fit 好的 scaler
                scaler_c=self.scaler_c  # 传入 fit 好的 scaler
            )
            test_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(test_features).float(),
                torch.from_numpy(test_targets).float(),
                torch.from_numpy(test_original_end_indices).long() # 添加原始结束索引
            )
            self.test_datasets.append(test_dataset)
            
        print(f"✅ 数据准备完成 (2特征，已应用全局标准化)")
        # === 全局标准化修改点4 结束 ===

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
