import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# # 重新导入需要的辅助函数 (如果这些函数在外部文件，请确保路径正确，或者直接集成)
# from ultra_precision_true_sota_25degC_copy import setup_sci_style, smooth_data

# 辅助函数 (已直接集成或修改)
def setup_sci_style():
    """设置matplotlib的SCI绘图风格，支持中文显示。"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # 启用中文支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def smooth_data(data, window_length, polyorder=2):
    """使用Savitzky-Golay滤波器平滑数据。"""
    # 确保window_length是奇数，并且小于数据长度
    if window_length % 2 == 0:
        window_length += 1
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def load_comparison_data_from_csv(csv_path):
    """
    从指定的CSV文件中加载对比数据。
    返回 Times, SOC_TRUE, SOC_Pred, SOE_TRUE, SOE_Pred
    所有数据已乘以100转换为百分比。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件未找到: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 确保列名正确
    required_cols = ['Times', 'SOC_TRUE', 'SOC_Pred', 'SOE_TRUE', 'SOE_Pred']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col}")

    times = df['Times'].values
    soc_true = df['SOC_TRUE'].values * 100
    soc_pred = df['SOC_Pred'].values * 100
    soe_true = df['SOE_TRUE'].values * 100
    soe_pred = df['SOE_Pred'].values * 100
    
    return times, soc_true, soc_pred, soe_true, soe_pred

# --- 辅助函数 ---
# 设置SCI绘图样式
setup_sci_style()

# --- 模型配置 (更新为直接的CSV路径) ---
MODELS_CONFIG = {
    "EKAN-T": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\实验数据\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\实验数据\UDDS_predictions.csv",
    },
    "TabPFN": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\tabpfn_enhanced_LA92_test_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\tabpfn_enhanced_LA92_test_results\raw_predictions\UDDS_predictions.csv",
    },
    "TCN-LSTM": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_tcn_lstm_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_tcn_lstm_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    },
    "LSTM": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_lstm_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_lstm_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    },
    "BiGRU": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_bigru_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_bigru_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    },
    "Informer": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_informer_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_informer_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    },
    "N-BEATS": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_nbeats_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_nbeats_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    },
    "CNN-GRU": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_cnn_gru_2features_25degC_results\raw_predictions\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\baseline_cnn_gru_2features_25degC_results\raw_predictions\UDDS_predictions.csv",
    }
}

def main():
    parser = argparse.ArgumentParser(description='Generate Comparison Plots for SOC/SOE Prediction')
    parser.add_argument('--output_dir', type=str, default='comparison_plots', help='Directory to save generated plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting comparison plot generation...")
    
    # 存储所有模型和数据集的数据
    all_loaded_data = {} # {'Model_Dataset': {'Times': ..., 'SOC_TRUE': ..., 'SOC_Pred': ..., 'SOE_TRUE': ..., 'SOE_Pred': ...}}

    print("⏳ Loading prediction data from CSV files for all models...")

    for model_name, config in MODELS_CONFIG.items():
        print(f"   Loading predictions for {model_name}...")
        
        # 加载 LA92 数据
        try:
            times_la92, soc_true_la92, soc_pred_la92, soe_true_la92, soe_pred_la92 = load_comparison_data_from_csv(config["LA92_csv"])
            all_loaded_data[f"{model_name}_LA92"] = {
                'Times': times_la92,
                'SOC_TRUE': soc_true_la92,
                'SOC_Pred': soc_pred_la92,
                'SOE_TRUE': soe_true_la92,
                'SOE_Pred': soe_pred_la92
            }
        except FileNotFoundError as e:
            print(f"   跳过 {model_name} LA92: {e}")
            # continue # Don't continue, try to load UDDS even if LA92 failed
        except KeyError as e:
            print(f"   跳过 {model_name} LA92: {e}")
            # continue

        # 加载 UDDS 数据
        try:
            times_udds, soc_true_udds, soc_pred_udds, soe_true_udds, soe_pred_udds = load_comparison_data_from_csv(config["UDDS_csv"])
            all_loaded_data[f"{model_name}_UDDS"] = {
                'Times': times_udds,
                'SOC_TRUE': soc_true_udds,
                'SOC_Pred': soc_pred_udds,
                'SOE_TRUE': soe_true_udds,
                'SOE_Pred': soe_pred_udds
            }
        except FileNotFoundError as e:
            print(f"   跳过 {model_name} UDDS: {e}")
            continue
        except KeyError as e:
            print(f"   跳过 {model_name} UDDS: {e}")
            continue

    gc.collect()

    # 定义用户提供的平均RMSE和平均MAE值
    user_provided_avg_rmse = {
        "N-BEATS": 2.667,
        "CNN-GRU": 2.701,
        "LSTM": 2.485,
        "TCN-LSTM": 2.493,
        "BiGRU": 2.312,
        "Informer": 2.664,
        "TabPFN": 1.625,
        "EKAN-T": 1.417
    }

    user_provided_avg_mae = {
        "N-BEATS": 2.056,
        "CNN-GRU": 2.127,
        "LSTM": 1.981,
        "TCN-LSTM": 1.971,
        "BiGRU": 1.792,
        "Informer": 2.144,
        "TabPFN": 1.286,
        "EKAN-T": 1.166
    }

    # --- 1. 生成时间序列对比图 ---
    print("🎨 Generating time series comparison plots...")
    
    # 选择用于时间序列图的所有模型
    selected_models_for_ts_plot = list(MODELS_CONFIG.keys()) # 所有模型

    fig = plt.figure(figsize=(18, 9)) # 调整整体图大小，长宽比2:1
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3) # 调整为4行2列
    
    # 主标题移到顶部中央
    fig.suptitle('Prediction and Error Comparison Across Models and Drive Cycles', 
                 fontsize=16, fontweight='bold', y=0.98) # 更通用的主标题

    # 定义8个轴对象
    axes = {
        'LA92_SOC_Pred': fig.add_subplot(gs[0, 0]),
        'LA92_SOC_Error': fig.add_subplot(gs[0, 1]),
        'LA92_SOE_Pred': fig.add_subplot(gs[1, 0]),
        'LA92_SOE_Error': fig.add_subplot(gs[1, 1]),
        'UDDS_SOC_Pred': fig.add_subplot(gs[2, 0]),
        'UDDS_SOC_Error': fig.add_subplot(gs[2, 1]),
        'UDDS_SOE_Pred': fig.add_subplot(gs[3, 0]),
        'UDDS_SOE_Error': fig.add_subplot(gs[3, 1]),
    }

    subplot_labels = iter(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']) # 使用迭代器生成标签

    # 颜色循环 (使用 tab10 调色板，确保有足够的区分度)
    colors = plt.cycler('color', plt.cm.tab10.colors) 

    datasets_to_plot = ["LA92", "UDDS"]
    metrics_to_plot = [("SOC", "SOC Prediction", "SOC Error (%)", "SOC (%)"), 
                       ("SOE", "SOE Prediction", "SOE Error (%)", "SOE (%)")]

    for dataset_type in datasets_to_plot:
        # 确保 EKAN-T 数据已加载 (作为真实值基准)
        if f"EKAN-T_{dataset_type}" not in all_loaded_data:
            print(f"Warning: EKAN-T {dataset_type} data not found, skipping plots for {dataset_type} in combined figure.")
            continue # Skip this dataset if EKAN-T data is missing

        ekant_data = all_loaded_data[f"EKAN-T_{dataset_type}"]
        ekant_times = ekant_data['Times']
        actual_data_len = len(ekant_times)

        for metric_prefix, pred_title, err_ylabel, pred_ylabel in metrics_to_plot:
            current_ax_pred = axes[f'{dataset_type}_{metric_prefix}_Pred']
            current_ax_err = axes[f'{dataset_type}_{metric_prefix}_Error']

            ekant_true_metric = ekant_data[f'{metric_prefix}_TRUE']

            # --- 绘制预测图 ---
            current_ax_pred.plot(ekant_times, ekant_true_metric, color='black', linewidth=2.0, label='Actual Value', alpha=0.8)
            current_ax_pred.set_prop_cycle(colors) # 重置颜色循环

            for model_name in selected_models_for_ts_plot:
                if f"{model_name}_{dataset_type}" in all_loaded_data:
                    model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                    pred_metric = model_data[f'{metric_prefix}_Pred']
                    current_true_metric = model_data[f'{metric_prefix}_TRUE']
                    current_times = model_data['Times']
                    
                    current_ax_pred.plot(current_times, smooth_data(pred_metric, window_length=501), linewidth=1.2, label=model_name) # 调整预测曲线线宽
                    
                    # --- 绘制误差图 ---
                    error_metric = smooth_data(pred_metric, window_length=77) - smooth_data(current_true_metric, window_length=77)
                    current_ax_err.plot(current_times, error_metric, linewidth=1.2, alpha=0.8, label=model_name) # 调整误差曲线线宽
                else:
                    print(f"Warning: {model_name} {dataset_type} data not found or invalid, skipping for {metric_prefix} plot.")
            
            # --- 预测图设置 ---
            current_ax_pred.set_title(f'{next(subplot_labels)} {dataset_type} {pred_title}', fontsize=11, fontweight='normal')
            current_ax_pred.set_ylabel(pred_ylabel, fontsize=10)
            current_ax_pred.legend(loc='upper right', fontsize=8, frameon=False, ncol=2) # 图例分2列
            current_ax_pred.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            current_ax_pred.set_ylim(0, 100)

            # --- 误差图设置 ---
            current_ax_err.set_title(f'{next(subplot_labels)} {dataset_type} {metric_prefix} Error', fontsize=11, fontweight='normal')
            current_ax_err.set_ylabel(err_ylabel, fontsize=10)
            current_ax_err.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.8)
            current_ax_err.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            current_ax_err.set_ylim(-8, 8)

            # --- 放大子图 ---
            axins = None # Reset axins for each plot
            
            zoom_start_default = 0
            zoom_length = 800 # Default zoom window length

            if dataset_type == "LA92":
                zoom_start_default = 2000 # Earlier for LA92
                zoom_length = 1000 # A bit longer for LA92 zoom
            elif dataset_type == "UDDS":
                zoom_start_default = 10000 # Later for UDDS
                zoom_length = 800

            zoom_start = zoom_start_default
            zoom_end = min(zoom_start + zoom_length, actual_data_len)

            if zoom_end <= zoom_start:
                print(f"Warning: Cannot create {metric_prefix} zoom for {dataset_type} due to insufficient data length. Skipping inset.")
            else:
                axins = current_ax_pred.inset_axes([0.08, 0.08, 0.4, 0.4], transform=current_ax_pred.transAxes) # Adjust position to bottom left
                y_min_zoom = 100
                y_max_zoom = 0

                true_zoom_segment = ekant_true_metric[ekant_times.searchsorted(zoom_start):ekant_times.searchsorted(zoom_end)]
                if len(true_zoom_segment) > 0:
                    y_min_zoom = min(y_min_zoom, np.min(true_zoom_segment))
                    y_max_zoom = max(y_max_zoom, np.max(true_zoom_segment))

                for model_name in selected_models_for_ts_plot:
                    if f"{model_name}_{dataset_type}" in all_loaded_data:
                        model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                        pred_metric = model_data[f'{metric_prefix}_Pred']
                        current_times = model_data['Times']

                        start_idx = current_times.searchsorted(zoom_start)
                        end_idx = current_times.searchsorted(zoom_end)
                        
                        pred_metric_zoom = pred_metric[start_idx:end_idx]
                        if len(pred_metric_zoom) > 0:
                            y_min_zoom = min(y_min_zoom, np.min(smooth_data(pred_metric_zoom, window_length=501)))
                            y_max_zoom = max(y_max_zoom, np.max(smooth_data(pred_metric_zoom, window_length=501)))

                y_range_zoom = y_max_zoom - y_min_zoom
                if y_range_zoom == 0: y_range_zoom = 1

                axins.plot(ekant_times[ekant_times.searchsorted(zoom_start):ekant_times.searchsorted(zoom_end)], 
                           true_zoom_segment, color='black', linewidth=2.0, label='Actual Value', alpha=0.8)
                axins.set_prop_cycle(colors)
                for model_name in selected_models_for_ts_plot:
                    if f"{model_name}_{dataset_type}" in all_loaded_data:
                        model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                        pred_metric = model_data[f'{metric_prefix}_Pred']
                        current_times = model_data['Times']

                        start_idx = current_times.searchsorted(zoom_start)
                        end_idx = current_times.searchsorted(zoom_end)
                        axins.plot(current_times[start_idx:end_idx], smooth_data(pred_metric[start_idx:end_idx], window_length=501), linewidth=1.2, label=model_name) # 调整预测曲线线宽
                
                axins.set_xlim(zoom_start, zoom_end)
                axins.set_ylim(y_min_zoom - 0.05 * y_range_zoom, y_max_zoom + 0.05 * y_range_zoom)
                axins.set_xticks([])
                axins.set_yticks([])
                axins.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                current_ax_pred.indicate_inset_zoom(axins, edgecolor='darkred', linewidth=1.0)
    
    # 为整个 figure 添加 X 轴标签
    fig.text(0.5, 0.03, 'Time(s)', ha='center', va='center', fontsize=11)

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.96]) # 调整布局，为底部标题和顶部主标题留出空间
    plt.savefig(os.path.join(args.output_dir, f'Figure_Time_Series_Comparison_All_Datasets.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png', pad_inches=0.1)
    plt.close(fig) # 关闭当前图，避免内存泄漏
    print(f"   Saved Figure_Time_Series_Comparison_All_Datasets.png")

    # --- 2. 生成RMSE和MAE对比柱状图 ---
    print("📊 Generating RMSE and MAE comparison bar chart...")
    
    # 用户提供的所有模型及其分工况的RMSE和MAE数据
    model_names_ordered = ["N-BEATS", "CNN-GRU", "LSTM", "TCN-LSTM", "BiGRU", "Informer", "TabPFN", "EKAN-T"]
    
    # 注意：这里的数据顺序必须与model_names_ordered一致
    la92_rmses = [1.9865, 1.734, 2.024, 2.066, 2.065, 2.155, 2.193, 1.547]
    udds_rmses = [1.5405, 1.1375, 1.0685, 1.227, 1.0765, 1.3015, 1.0555, 1.0195]
    la92_maes = [1.5775, 1.423, 1.6925, 1.5805, 1.663, 1.816, 1.7815, 1.334]
    udds_maes = [1.198, 0.895, 0.834, 0.937, 0.8585, 0.9695, 0.7915, 0.774]

    # 构建用于柱状图的DataFrame
    plot_data = []
    for i, model_name in enumerate(model_names_ordered):
        # 计算平均RMSE和MAE
        avg_rmse_calculated = (la92_rmses[i] + udds_rmses[i]) / 2
        avg_mae_calculated = (la92_maes[i] + udds_maes[i]) / 2
        
        plot_data.append({'Model': model_name, 'Dataset': 'LA92', 'RMSE': la92_rmses[i], 'MAE': la92_maes[i], 
                          'Average_RMSE': avg_rmse_calculated, 'Average_MAE': avg_mae_calculated})
        plot_data.append({'Model': model_name, 'Dataset': 'UDDS', 'RMSE': udds_rmses[i], 'MAE': udds_maes[i], 
                          'Average_RMSE': avg_rmse_calculated, 'Average_MAE': avg_mae_calculated})
    
    results_df_for_plot = pd.DataFrame(plot_data)


    # 绘制柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # RMSE 柱状图
    ax_rmse = axes[0]
    # 对模型按计算出的平均RMSE排序，并将EKAN-T放在最前面
    other_models_for_sort = [m for m in model_names_ordered if m != "EKAN-T"]
    sorted_other_models = sorted(other_models_for_sort, key=lambda x: results_df_for_plot[results_df_for_plot['Model']==x]['Average_RMSE'].iloc[0])
    sorted_models = ["EKAN-T"] + sorted_other_models

    bar_width = 0.35
    index = np.arange(len(sorted_models))

    # LA92 RMSE
    la92_rmse_data = [results_df_for_plot[(results_df_for_plot['Model'] == m) & (results_df_for_plot['Dataset'] == 'LA92')]['RMSE'].iloc[0] for m in sorted_models]
    ax_rmse.bar(index, la92_rmse_data, bar_width, label='LA92 RMSE', color='#0072BD')

    # UDDS RMSE
    udds_rmse_data = [results_df_for_plot[(results_df_for_plot['Model'] == m) & (results_df_for_plot['Dataset'] == 'UDDS')]['RMSE'].iloc[0] for m in sorted_models]
    ax_rmse.bar(index + bar_width, udds_rmse_data, bar_width, label='UDDS RMSE', color='#D95319')

    # 平均RMSE (RMSĒ) - 直接使用计算出的平均值
    avg_rmse_data_plot = [results_df_for_plot[results_df_for_plot['Model'] == m]['Average_RMSE'].iloc[0] for m in sorted_models]
    ax_rmse.plot(index + bar_width/2, avg_rmse_data_plot, 'o--', color='black', label='Average RMSE') # 移除上标


    ax_rmse.set_xlabel('Model', fontsize=11)
    ax_rmse.set_ylabel('RMSE', fontsize=11)
    ax_rmse.set_title('Comprehensive RMSE Comparison (SOC & SOE Average) Across Models and Drive Cycles', fontsize=12, fontweight='normal') # 更清晰的标题
    ax_rmse.set_xticks(index + bar_width / 2)
    ax_rmse.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
    ax_rmse.legend(fontsize=9, frameon=False, handlelength=1.5) # 调整handlelength
    ax_rmse.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax_rmse.set_ylim(bottom=0) # 从0开始

    # 在RMSE柱状图顶部标注用户提供的平均RMSĒ的值
    for i, model_name in enumerate(sorted_models):
        avg_rmse_val = results_df_for_plot[results_df_for_plot['Model'] == model_name]['Average_RMSE'].iloc[0]
        ax_rmse.text(index[i] + bar_width / 2, avg_rmse_val + 0.05, f'{avg_rmse_val:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='black') # 调整标注位置

    # MAE 柱状图
    ax_mae = axes[1]
    # LA92 MAE
    la92_mae_data = [results_df_for_plot[(results_df_for_plot['Model'] == m) & (results_df_for_plot['Dataset'] == 'LA92')]['MAE'].iloc[0] for m in sorted_models]
    ax_mae.bar(index, la92_mae_data, bar_width, label='LA92 MAE', color='#0072BD')

    # UDDS MAE
    udds_mae_data = [results_df_for_plot[(results_df_for_plot['Model'] == m) & (results_df_for_plot['Dataset'] == 'UDDS')]['MAE'].iloc[0] for m in sorted_models]
    ax_mae.bar(index + bar_width, udds_mae_data, bar_width, label='UDDS MAE', color='#D95319')

    # 平均MAE (MAĒ) - 直接使用计算出的平均值
    avg_mae_data_plot = [results_df_for_plot[results_df_for_plot['Model'] == m]['Average_MAE'].iloc[0] for m in sorted_models]
    ax_mae.plot(index + bar_width/2, avg_mae_data_plot, 'o--', color='black', label='Average MAE') # 移除上标

    ax_mae.set_xlabel('Model', fontsize=11)
    ax_mae.set_ylabel('MAE', fontsize=11)
    ax_mae.set_title('Comprehensive MAE Comparison (SOC & SOE Average) Across Models and Drive Cycles', fontsize=12, fontweight='normal') # 更清晰的标题
    ax_mae.set_xticks(index + bar_width / 2)
    ax_mae.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
    ax_mae.legend(fontsize=9, frameon=False, handlelength=1.5) # 调整handlelength
    ax_mae.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax_mae.set_ylim(bottom=0) # 从0开始

    # 在MAE柱状图顶部标注用户提供的平均MAĒ的值
    for i, model_name in enumerate(sorted_models):
        avg_mae_val = results_df_for_plot[results_df_for_plot['Model'] == model_name]['Average_MAE'].iloc[0]
        ax_mae.text(index[i] + bar_width / 2, avg_mae_val + 0.05, f'{avg_mae_val:.3f}', 
                   ha='center', va='bottom', fontsize=8, color='black') # 调整标注位置

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'Figure_RMSE_MAE_Comparison_Bar_Chart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png', pad_inches=0.1)
    plt.close(fig)
    print("   Saved Figure_RMSE_MAE_Comparison_Bar_Chart.png")

    print("✅ Comparison plot generation complete.")

if __name__ == '__main__':
    main()
