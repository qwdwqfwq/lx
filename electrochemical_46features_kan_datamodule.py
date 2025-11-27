import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl

# 导入原始的特征工程模块
from electrochemical_features import ElectrochemicalFeatureEngineer


def _infer_workload_label(file_path: str) -> int:
    """根据文件名推断工况标签: NN->0, UDDS->1, 其他->-1(无标签)"""
    fp_lower = file_path.lower()
    if "udds" in fp_lower:
        return 1
    if "nn" in fp_lower:
        return 0
    return -1


def process_single_dataframe_for_46features(
    data_df: pd.DataFrame,
    temperature,
    engineer: ElectrochemicalFeatureEngineer,
    window_size=32,
    overlap_ratio=0.5,
    is_test_set=False,
    feature_scalers=None,
    # workload_label: int = -1, # 移除 workload_label 参数
):
    """
    对单个原始DataFrame进行46个电化学特征工程，并进行窗口化处理。
    支持全局标准化、测试集步长为1，不再返回工况标签。
    """
    all_features_list, all_targets_list, all_original_end_indices = [], [], [] # 移除 all_labels_list

    # 确定步长
    step_size = 1 if is_test_set else max(1, int(window_size * (1 - overlap_ratio)))

    try:
        if data_df.empty:
            print(f"   输入DataFrame为空，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        # --- 特征工程 ---
        features_df = engineer.create_electrochemical_features(data_df, temperature)
        targets_df = data_df[["SOC", "SOE"]]

        if features_df.empty or targets_df.empty:
            print(f"   特征或目标为空，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        # --- 特征标准化 ---
        if feature_scalers:
            if is_test_set is False and feature_scalers == "fit_only":
                scaled_features = features_df.values
            else:  # 应用标准化
                if not hasattr(feature_scalers, "transform"):
                    raise ValueError("Provided feature_scalers must be a fitted MinMaxScaler instance.")
                scaled_features = feature_scalers.transform(features_df.values)
        else:
            # 未提供 scaler，不进行标准化
            scaled_features = features_df.values

        # --- 目标不标准化，保持原始物理量 ---
        scaled_targets = targets_df.values

        # --- 滑动窗口处理 ---
        max_i = len(scaled_features) - window_size
        if max_i < 0:
            print(f"   数据长度不足以创建窗口，跳过。")
            return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

        for i in range(0, max_i + 1, step_size):
            window_features = scaled_features[i : i + window_size]
            target_value = scaled_targets[i + window_size - 1]  # 预测窗口的最后一个时间步
            if not (np.isnan(window_features).any() or np.isnan(target_value).any()):
                all_features_list.append(window_features)
                all_targets_list.append(target_value)
                all_original_end_indices.append(i + window_size - 1)
                # all_labels_list.append(workload_label) # 移除标签添加
    except Exception as e:
        print(f"   处理DataFrame时发生错误: {e}")
        import traceback

        traceback.print_exc()
        return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

    if not all_features_list:
        print(f"   未能从提供的DataFrame中加载到任何有效数据。")
        return np.array([]), np.array([]), np.array([]) # 确保返回3个空数组

    return (
        np.stack(all_features_list, dtype=np.float32),
        np.stack(all_targets_list, dtype=np.float32),
        np.stack(all_original_end_indices, dtype=np.int64),
        # np.stack(all_labels_list, dtype=np.int64), # 移除标签返回
    )


def _load_all_with_labels(csv_paths, temperature, engineer, window_size, overlap_ratio, is_test_set, scaler):
    feat_list, tgt_list, idx_list = [], [], [] # 移除 label_list
    for file_path in csv_paths:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"   加载文件 {file_path} 时发生错误: {e}")
            continue
        # label = _infer_workload_label(file_path) # 移除标签推断
        f, t, idx = process_single_dataframe_for_46features(
            df,
            temperature,
            engineer,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            is_test_set=is_test_set,
            feature_scalers=scaler,
            # workload_label=label, # 移除 workload_label 传递
        )
        if f.size == 0:
            continue
        feat_list.append(f)
        tgt_list.append(t)
        idx_list.append(idx)
        # label_list.append(lbl) # 移除标签添加

    if not feat_list:
        return np.array([]), np.array([]), np.array([]) # 移除标签返回

    return (
        np.concatenate(feat_list, axis=0),
        np.concatenate(tgt_list, axis=0),
        np.concatenate(idx_list, axis=0),
        # np.concatenate(label_list, axis=0), # 移除标签连接
    )


class Electrochemical46FeaturesKANDataModule(pl.LightningDataModule):
    """
    电化学46特征数据模块，支持全局MinMaxScaler、测试集步长为1，不再返回工况标签。
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.feature_scaler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = []
        self.engineer = ElectrochemicalFeatureEngineer()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"开始设置电化学46特征KAN数据模块 (温度: {self.hparams.temperature}°C, 阶段: {stage})")

        # 1) 拟合特征scaler
        if stage == "fit" and self.feature_scaler is None:
            print("   收集训练数据以拟合全局MinMaxScaler...")
            raw_features, raw_targets, raw_idx = _load_all_with_labels(
                self.hparams.train_paths,
                self.hparams.temperature,
                self.engineer,
                self.hparams.window_size,
                self.hparams.overlap_ratio,
                is_test_set=False,
                scaler="fit_only",
            ) # 移除 raw_labels
            if raw_features.size == 0:
                raise ValueError("训练集数据为空，无法进行标准化器拟合。请检查train_paths。")
            flattened = raw_features.reshape(-1, raw_features.shape[-1])
            self.feature_scaler = MinMaxScaler()
            self.feature_scaler.fit(flattened)
            print("   全局特征MinMaxScaler拟合完成。")
            # 缓存原始数据
            self.cached_raw_train_features = raw_features
            self.cached_raw_train_targets = raw_targets
            self.cached_raw_train_original_end_indices = raw_idx
            # self.cached_raw_train_labels = raw_labels # 移除标签缓存

        # 2) 加载&标准化训练/验证数据
        print("   加载和处理训练/验证数据...")
        if self.feature_scaler is not None and hasattr(self, "cached_raw_train_features"):
            train_features_scaled = self.feature_scaler.transform(
                self.cached_raw_train_features.reshape(-1, self.cached_raw_train_features.shape[-1])
            ).reshape(self.cached_raw_train_features.shape)
            train_features = train_features_scaled
            train_targets = self.cached_raw_train_targets
            train_original_end_indices = self.cached_raw_train_original_end_indices
            # train_labels = self.cached_raw_train_labels # 移除标签获取
        else:
            print("   警告: 未能利用缓存的原始训练数据，重新执行特征工程和MinMaxScaler。")
            train_features, train_targets, train_original_end_indices = _load_all_with_labels(
                self.hparams.train_paths,
                self.hparams.temperature,
                self.engineer,
                self.hparams.window_size,
                self.hparams.overlap_ratio,
                is_test_set=False,
                scaler=self.feature_scaler,
            ) # 移除 train_labels

        if train_features.size == 0:
            raise ValueError("训练/验证数据加载失败，请检查train_paths。")

        X_train_tensor = torch.from_numpy(train_features).float()
        y_train_tensor = torch.from_numpy(train_targets).float()
        train_original_end_indices_tensor = torch.from_numpy(train_original_end_indices).long()
        # train_labels_tensor = torch.from_numpy(train_labels).long() # 移除标签张量创建

        # 数据划分
        dataset_size = len(X_train_tensor)
        if dataset_size == 0:
            raise ValueError("加载到的训练集样本数量为0，请检查数据。")

        train_size = int(self.hparams.train_val_split_ratio * dataset_size)
        val_size = dataset_size - train_size
        if train_size == 0:
            train_size = 1
            val_size = max(dataset_size - 1, 0)
        elif val_size == 0 and dataset_size > 1:
            val_size = 1
            train_size = dataset_size - 1

        print(f"   数据划分 (训练集比例: {self.hparams.train_val_split_ratio*100:.0f}%)： 训练={train_size:,}, 验证={val_size:,}")

        full_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_original_end_indices_tensor) # 移除 train_labels_tensor
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.hparams.seed)
        )

        # 3) 处理测试数据（步长=1，逐文件保留标签）
        print("   加载和处理测试数据 (步长=1)...")
        self.test_datasets = []
        for file_path in self.hparams.test_paths:
            # label_guess = _infer_workload_label(file_path) # 移除标签推断
            df = pd.read_csv(file_path)
            test_features, test_targets, test_original_end_indices = process_single_dataframe_for_46features(
                df,
                self.hparams.temperature,
                self.engineer,
                window_size=self.hparams.window_size,
                overlap_ratio=self.hparams.overlap_ratio,
                is_test_set=True,
                feature_scalers=self.feature_scaler,
                # workload_label=label_guess, # 移除 workload_label 传递
            ) # 移除 test_labels
            if test_features.size == 0:
                print(f"   测试文件 {file_path.split('/')[-1]} 加载失败或为空，跳过。")
                continue
            test_dataset = TensorDataset(
                torch.from_numpy(test_features).float(),
                torch.from_numpy(test_targets).float(),
                torch.from_numpy(test_original_end_indices).long(),
                # torch.from_numpy(test_labels).long(), # 移除标签张量创建
            )
            self.test_datasets.append(test_dataset)
        if not self.test_datasets:
            print("   测试数据为空，未创建测试DataLoader。")

        print(f"电化学46特征数据模块设置完成，特征数: {train_features.shape[-1]}")

    @property
    def scaler(self):
        if self.feature_scaler is None:
            raise RuntimeError("特征标准化器尚未拟合。请先运行 setup(stage='fit')。")
        return {"features": self.feature_scaler}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
            for ds in self.test_datasets
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Electrochemical 46 Features DataModule Test")
    parser.add_argument(
        "--train_paths",
        type=str,
        nargs="+",
        default=[
            r"C:\25degC training\03-18-17_02.17 25degC_Cycle_1_Pan18650PF.csv",
            r"C:\25degC training\03-19-17_03.25 25degC_Cycle_2_Pan18650PF.csv",
        ],
    )
    parser.add_argument(
        "--test_paths",
        type=str,
        nargs="+",
        default=[
            r"C:\25degC testing\03-21-17_00.29 25degC_UDDS_Pan18650PF.csv",
        ],
    )
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--output_features", type=str, default="SOC,SOE")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_val_split_ratio", type=float, default=0.93)

    args = parser.parse_args()
    args.output_features = [item.strip() for item in args.output_features.split(",")]

    print("测试 Electrochemical46FeaturesKANDataModule...")

    datamodule = Electrochemical46FeaturesKANDataModule(args)
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loaders = datamodule.test_dataloader()

    print(f"训练数据加载器批次数量: {len(train_loader)}")
    print(f"验证数据加载器批次数量: {len(val_loader)}")
    print(f"测试数据加载器数量: {len(test_loaders)}")

    if train_loader:
        for batch_idx, batch in enumerate(train_loader):
            x, y, original_indices = batch # 修改解包为3个元素
            print(f"训练批次 {batch_idx}: X形状={x.shape}, Y形状={y.shape}, 原始索引形状={original_indices.shape}") # 移除标签形状
            break

    if test_loaders:
        for ds_idx, test_loader in enumerate(test_loaders):
            for batch_idx, batch in enumerate(test_loader):
                x, y, original_indices = batch # 修改解包为3个元素
                print(f"测试集{ds_idx} 批次 {batch_idx}: X形状={x.shape}, Y形状={y.shape}, 原始索引形状={original_indices.shape}") # 移除标签形状
                break
            break

    print("Electrochemical46FeaturesKANDataModule 测试完成。")
