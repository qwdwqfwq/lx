import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl

# 导入原始的特征工程模块
from electrochemical_features import ElectrochemicalFeatureEngineer

def process_single_dataframe_for_46features(
    data_df: pd.DataFrame, 
    temperature, 
    engineer: ElectrochemicalFeatureEngineer, 
    window_size=32, 
    overlap_ratio=0.5, 
    is_test_set=False,
    feature_scalers=None
):
    """
    对单个原始DataFrame进行46个电化学特征工程，并进行窗口化处理。
    支持全局标准化和测试集步长为1。
    """
    all_features_list, all_targets_list, all_original_end_indices = [], [], []
    
    # 确定步长
    step_size = 1 if is_test_set else max(1, int(window_size * (1 - overlap_ratio)))

    try:
        if data_df.empty:
            print(f"   ⚠️ 输入DataFrame为空，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        # --- 特征工程 ---
        features_df = engineer.create_electrochemical_features(data_df, temperature)
        targets_df = data_df[['SOC', 'SOE']]

        if features_df.empty or targets_df.empty:
            print(f"   ⚠️ 特征或目标为空，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        # --- 特征标准化 ---
        if feature_scalers:
            if is_test_set is False and feature_scalers == 'fit_only':
                scaled_features = features_df.values
            else: # 应用标准化
                if not hasattr(feature_scalers, 'transform'):
                    raise ValueError("Provided feature_scalers must be a fitted StandardScaler instance.")
                scaled_features = feature_scalers.transform(features_df.values)
        else:
            # 没有提供 scaler，不进行标准化 (仅用于拟合阶段获取原始数据)
            scaled_features = features_df.values

        # --- 目标：不进行标准化 (保持原始物理值) ---
        scaled_targets = targets_df.values # 直接使用原始物理值

        # --- 滑动窗口处理 ---
        max_i = len(scaled_features) - window_size
        if max_i < 0:
            print(f"   ⚠️ 数据长度不足以创建窗口，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        for i in range(0, max_i + 1, step_size):
            window_features = scaled_features[i : i + window_size]
            target_value = scaled_targets[i + window_size - 1] # 预测窗口的最后一个时间步，现在是原始物理值

            if not (np.isnan(window_features).any() or np.isnan(target_value).any()):
                all_features_list.append(window_features)
                all_targets_list.append(target_value)
                all_original_end_indices.append(i + window_size - 1) # 记录原始时间步索引

    except Exception as e:
        print(f"   ❌ 处理DataFrame时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

    if not all_features_list:
        print(f"   ⚠️ 未能从提供的DataFrame中加载到任何有效数据。")
        return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组
    
    # 🚨 [优化] 使用 np.stack 替代 np.array，可能更高效
    return np.stack(all_features_list, dtype=np.float32), \
           np.stack(all_targets_list, dtype=np.float32), \
           np.stack(all_original_end_indices, dtype=np.int64) # 索引用int64确保不会溢出

def _load_data_from_file_paths(
    csv_paths,
    temperature,
    engineer,
    window_size,
    overlap_ratio,
    is_test_set,
    scaler,
):
    feat_list, tgt_list, idx_list = [], [], []
    for file_path in csv_paths:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"   加载文件 {file_path.split('/')[-1]} 时发生错误: {e}")
            continue
        f, t, idx = process_single_dataframe_for_46features(
            df,
            temperature,
            engineer,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            is_test_set=is_test_set,
            feature_scalers=scaler,
        )
        if f.size == 0:
            continue
        feat_list.append(f)
        tgt_list.append(t)
        idx_list.append(idx)

    if not feat_list:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(feat_list, axis=0),
        np.concatenate(tgt_list, axis=0),
        np.concatenate(idx_list, axis=0),
    )


class Electrochemical46FeaturesDataModule(pl.LightningDataModule):
    """
    电化学46特征数据模块，支持全局标准化和测试集步长为1。
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.feature_scaler = None # 用于存储拟合后的46特征全局标准化器
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = []
        self.engineer = ElectrochemicalFeatureEngineer() # 只初始化一次特征工程师

    def prepare_data(self):
        # 在多GPU场景下，此方法只在一个进程上运行，适合下载数据等
        pass

    def setup(self, stage=None):
        print(f"🧪 开始设置电化学46特征数据模块 (温度: {self.hparams.temperature}°C, 阶段: {stage})")

        # --- 1. 基于文件进行训练集和验证集的划分 ---
        # 确保训练文件列表是可预测的顺序
        train_files_shuffled = sorted(self.hparams.train_paths) # 保持顺序一致性
        total_train_files = len(train_files_shuffled)
        
        # 计算训练文件数量，至少保留1个文件用于训练
        num_train_for_split = max(1, int(total_train_files * self.hparams.train_val_split_ratio))
        if num_train_for_split == total_train_files: # 确保至少有一个验证文件，如果文件数足够
            if total_train_files > 1:
                num_train_for_split = total_train_files - 1
            else:
                # 如果只有一个文件，就全部用于训练，验证集为空
                num_train_for_split = total_train_files

        train_files = train_files_shuffled[:num_train_for_split]
        val_files = train_files_shuffled[num_train_for_split:]

        if not train_files:
            raise ValueError("没有训练文件可用。请检查train_paths或train_val_split_ratio。")
        # 即使val_files为空，也允许继续，因为有些场景可能不需要验证集
        if not val_files:
            print("   警告: 没有验证文件可用。模型将在没有独立验证集的情况下进行训练。")

        print(f"   文件划分： 训练文件数={len(train_files)}, 验证文件数={len(val_files)}")

        # --- 2. 拟合特征scaler (仅在训练文件上) ---
        if stage == "fit" and self.feature_scaler is None:
            print("   📊 收集训练文件数据以拟合全局StandardScaler...")
            # 仅加载训练文件的数据用于Scaler拟合
            raw_train_features_for_scaler, _, _ = _load_data_from_file_paths(
                train_files,
                self.hparams.temperature,
                self.engineer,
                self.hparams.window_size,
                self.hparams.overlap_ratio,
                is_test_set=False,
                scaler="fit_only", # 标记为只用于拟合
            )
            if raw_train_features_for_scaler.size == 0:
                raise ValueError("训练集数据为空，无法进行标准化器拟合。请检查train_files。")
            
            # 将所有窗口的特征展平进行拟合
            flattened_train_features = raw_train_features_for_scaler.reshape(-1, raw_train_features_for_scaler.shape[-1])
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(flattened_train_features)
            print("   ✅ 全局特征StandardScaler拟合完成（仅基于训练文件）。")

        # --- 3. 加载和标准化训练数据 ---
        print("   📁 加载和处理训练数据...")
        train_features, train_targets, train_original_end_indices = _load_data_from_file_paths(
            train_files,
            self.hparams.temperature,
            self.engineer,
            self.hparams.window_size,
            self.hparams.overlap_ratio,
            is_test_set=False,
            feature_scalers=self.feature_scaler, # 应用拟合好的Scaler
        )
        if train_features.size == 0:
            raise ValueError("训练数据加载失败，请检查train_files。")

        X_train_tensor = torch.from_numpy(train_features).float()
        y_train_tensor = torch.from_numpy(train_targets).float()
        train_original_end_indices_tensor = torch.from_numpy(train_original_end_indices).long()
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_original_end_indices_tensor)
        print(f"   训练集样本数量: {len(self.train_dataset):,}")

        # --- 4. 加载和标准化验证数据 ---
        print("   📁 加载和处理验证数据...")
        if val_files:
            val_features, val_targets, val_original_end_indices = _load_data_from_file_paths(
                val_files,
                self.hparams.temperature,
                self.engineer,
                self.hparams.window_size,
                self.hparams.overlap_ratio,
                is_test_set=False,
                feature_scalers=self.feature_scaler, # 应用拟合好的Scaler
            )
            if val_features.size == 0:
                print("   ⚠️ 验证数据加载失败或为空。验证集将被设置为空。")
                self.val_dataset = None
            else:
                X_val_tensor = torch.from_numpy(val_features).float()
                y_val_tensor = torch.from_numpy(val_targets).float()
                val_original_end_indices_tensor = torch.from_numpy(val_original_end_indices).long()
                self.val_dataset = TensorDataset(X_val_tensor, y_val_tensor, val_original_end_indices_tensor)
                print(f"   验证集样本数量: {len(self.val_dataset):,}")
        else:
            self.val_dataset = None
            print("   未设置验证集，跳过验证数据加载。")

        # --- 5. 处理测试数据 (步长=1) ---
        print("   📁 加载和处理测试数据 (步长=1)...")
        self.test_datasets = []
        if self.hparams.test_paths:
            for file_path in self.hparams.test_paths:
                df = pd.read_csv(file_path)
                test_features, test_targets, test_original_end_indices = process_single_dataframe_for_46features(
                    df,
                    self.hparams.temperature,
                    self.engineer,
                    window_size=self.hparams.window_size,
                    overlap_ratio=self.hparams.overlap_ratio,
                    is_test_set=True,
                    feature_scalers=self.feature_scaler, # 应用拟合好的Scaler
                )
                if test_features.size == 0:
                    print(f"   ⚠️ 测试文件 {file_path.split('/')[-1]} 加载失败或为空，跳过。")
                    continue

                test_dataset = TensorDataset(
                    torch.from_numpy(test_features).float(),
                    torch.from_numpy(test_targets).float(),
                    torch.from_numpy(test_original_end_indices).long()
                )
                self.test_datasets.append(test_dataset)
            if not self.test_datasets:
                print("   测试数据为空，未创建测试DataLoader。")
        else:
            print("   未提供测试文件路径，跳过测试数据加载。")

        print(f"✅ 电化学46特征数据模块设置完成，训练特征数: {train_features.shape[-1]}")

    @property
    def scaler(self):
        """提供对已拟合的特征标准化器的访问"""
        if self.feature_scaler is None:
            raise RuntimeError("特征标准化器尚未拟合。请先运行 setup(stage='fit')。")
        return {'features': self.feature_scaler}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None # 如果没有验证集，返回None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return [DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        ) for ds in self.test_datasets]

if __name__ == "__main__":
    # 示例用法
    parser = argparse.ArgumentParser(description='Electrochemical 46 Features DataModule Test')
    parser.add_argument('--train_paths', type=str, nargs='+', default=[
        r"C:degC training-18-17_02.17 25degC_Cycle_1_Pan18650PF.csv",
        r"C:degC training-19-17_03.25 25degC_Cycle_2_Pan18650PF.csv",
    ])
    parser.add_argument('--test_paths', type=str, nargs='+', default=[
        r"C:degC testing-21-17_00.29 25degC_UDDS_Pan18650PF.csv",
    ])
    parser.add_argument('--temperature', type=float, default=25.0)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
    parser.add_argument('--output_features', type=str, default='SOC,SOE')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_val_split_ratio', type=float, default=0.93)
    
    args = parser.parse_args()
    args.output_features = [item.strip() for item in args.output_features.split(',')]

    print("🔬 测试 Electrochemical46FeaturesDataModule...")
    
    datamodule = Electrochemical46FeaturesDataModule(args)
    datamodule.setup(stage='fit')

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loaders = datamodule.test_dataloader()

    print(f"训练数据加载器批次数量: {len(train_loader)}")
    print(f"验证数据加载器批次数量: {len(val_loader)}")
    print(f"测试数据加载器数量: {len(test_loaders)}")

    if train_loader:
        for batch_idx, batch in enumerate(train_loader):
            x, y, original_indices = batch
            print(f"训练批次 {batch_idx}: X形状={x.shape}, Y形状={y.shape}, 原始索引形状={original_indices.shape}")
            break
    
    if test_loaders:
        for ds_idx, test_loader in enumerate(test_loaders):
            for batch_idx, batch in enumerate(test_loader):
                x, y, original_indices = batch
                print(f"测试集 {ds_idx} 批次 {batch_idx}: X形状={x.shape}, Y形状={y.shape}, 原始索引形状={original_indices.shape}")
                break
            break

    print("✅ Electrochemical46FeaturesDataModule 测试完成。")
