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

# # 重新导入需要的辅助函数 (如果这些函数在外部文件, 请确保路径正确, 或者直接集成)
# from ultra_precision_true_sota_25degC_copy import setup_sci_style, smooth_data

# 辅助函数 (已直接集成或修改)
def setup_sci_style():
    """设置matplotlib的SCI绘图风格, 支持中文显示。"""
    plt.style.use('default')
    
    # 字体设置 - 参照参考图
    plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'SimHei', 'DejaVu Sans'] # 遵循指令
    plt.rcParams['font.size'] = 11          # 遵循指令
    plt.rcParams['axes.titlesize'] = 12     # 遵循指令
    plt.rcParams['axes.labelsize'] = 11     # 遵循指令
    plt.rcParams['xtick.labelsize'] = 10    # 遵循指令
    plt.rcParams['ytick.labelsize'] = 10    # 遵循指令
    plt.rcParams['legend.fontsize'] = 10    # 遵循指令
    plt.rcParams['axes.unicode_minus'] = False # 遵循指令
    
    # SCI审美设置 - 模仿参考图
    plt.rcParams['axes.linewidth'] = 1.0    # 遵循指令
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.6
    plt.rcParams['ytick.minor.width'] = 0.6
    plt.rcParams['grid.linewidth'] = 0.5    # 非常细的网格线
    plt.rcParams['grid.alpha'] = 0.2        # 遵循指令, alpha=0.2
    plt.rcParams['lines.linewidth'] = 1.8   # 遵循指令
    
    # 移除 plt.rcParams['axes.prop_cycle'], 以便在绘图循环中动态管理颜色
    # plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
    #     ['#0072BD', '#D95319', '#4DBEEE', '#A2142F'])  # 蓝色、橙红色系

def smooth_data(data, window_length, polyorder=2):
    """使用Savitzky-Golay滤波器平滑数据。"""
    # 确保window_length是奇数, 并且小于数据长度
    if window_length % 2 == 0:
        window_length += 1
    if len(data) < window_length:
        return data
    
    # 确保polyorder < window_length
    polyorder = min(polyorder, window_length - 1)
    
    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception as e:
        print(f"Warning: Savitzky-Golay filter failed with window_length={window_length}, polyorder={polyorder}. Error: {e}. Returning original data.")
        return data

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
    required_cols = ['Times', 'SOC_Actual', 'SOC_Predicted', 'SOE_Actual', 'SOE_Predicted'] # 更新列名
    for col in required_cols:
        if col not in df.columns:
            # 尝试旧的列名兼容
            if col == 'SOC_Actual':
                if 'SOC_TRUE' in df.columns: df['SOC_Actual'] = df['SOC_TRUE']
                else: raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col} 或 SOC_TRUE")
            elif col == 'SOC_Predicted':
                if 'SOC_Pred' in df.columns: df['SOC_Predicted'] = df['SOC_Pred']
                else: raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col} 或 SOC_Pred")
            elif col == 'SOE_Actual':
                if 'SOE_TRUE' in df.columns: df['SOE_Actual'] = df['SOE_TRUE']
                else: raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col} 或 SOE_TRUE")
            elif col == 'SOE_Predicted':
                if 'SOE_Pred' in df.columns: df['SOE_Predicted'] = df['SOE_Pred']
                else: raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col} 或 SOE_Pred")
            else:
                 raise KeyError(f"CSV文件 '{csv_path}' 中缺少必需的列: {col}")

    times = df['Times'].values
    soc_true = df['SOC_Actual'].values * 100
    soc_pred = df['SOC_Predicted'].values * 100
    soe_true = df['SOE_Actual'].values * 100
    soe_pred = df['SOE_Predicted'].values * 100
    
    return times, soc_true, soc_pred, soe_true, soe_pred

# --- 辅助函数 ---
# 设置SCI绘图样式
setup_sci_style()

# --- 模型配置 (更新为直接的CSV路径) ---
MODELS_CONFIG = {
    "FE-KAN-T": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\实验数据\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\实验数据\UDDS_predictions.csv",
    },
    "No Workload Detector": {
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\基于KAN、KAN卷积的回归预测合集\=KAN+Transfomer时间序列预测\examples\\ablation_A1_1_no_detector_results\\raw_predictions\\LA92_predictions.csv",
        "UDDS_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A1_1_no_detector_results\\raw_predictions\\UDDS_predictions.csv",
    },
    "Only V-I Features": { 
        "LA92_csv": r"C:\Users\黎枭\Desktop\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A2_2_simple_features_results\\raw_predictions\\LA92_predictions.csv",
        "UDDS_csv": r"C:\\Users\\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A2_2_simple_features_results\\raw_predictions\\UDDS_predictions.csv",
    },
    "KAN": { # Renamed from "KAN Only"
        "LA92_csv": r"C:\Users\黎枭\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A3_2_kan_only_FIXED_results\\raw_predictions\\LA92_predictions.csv",
        "UDDS_csv": r"C:\\Users\\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A3_2_kan_only_FIXED_results\\raw_predictions\\UDDS_predictions.csv",
    },
    "Transformer": { # Renamed from "Transformer Only"
        "LA92_csv": r"C:\\Users\\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A3_1_std_transformer_results\\raw_predictions\\LA92_predictions.csv",
        "UDDS_csv": r"C:\\Users\\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A3_1_std_transformer_results\\raw_predictions\\UDDS_predictions.csv",
    },
    "Independent SOC/SOE": { # Added Independent SOC/SOE
        "LA92_csv": r"C:\Users\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A4_1_independent_MinMaxScaler_results\\raw_predictions\\LA92_predictions.csv",
        "UDDS_csv": r"C:\\Users\\黎枭\\Desktop\\旧电脑数据\\基于KAN、KAN卷积的回归预测合集\\=KAN+Transfomer时间序列预测\\examples\\ablation_A4_1_independent_MinMaxScaler_results\\raw_predictions\\UDDS_predictions.csv",
    },
}

def main():
    parser = argparse.ArgumentParser(description='Generate Comparison Plots for SOC/SOE Prediction')
    parser.add_argument('--output_dir', type=str, default='comparison_plots', help='Directory to save generated plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting comparison plot generation...")
    
    all_loaded_data = {} # {'Model_Dataset': {'Times': ..., 'SOC_TRUE': ..., 'SOC_Pred': ..., 'SOE_TRUE': ..., 'SOE_Pred': ...}}

    print("   Loading prediction data from CSV files for all models...")

    for model_name, config in MODELS_CONFIG.items():
        print(f"   Loading predictions for {model_name}...")
        
        for dataset_type in ["LA92", "UDDS"]:
            csv_path_key = f"{dataset_type}_csv"
            if csv_path_key in config and config[csv_path_key]: # 检查路径是否存在且不为空
                try:
                    times, soc_true, soc_pred, soe_true, soe_pred = load_comparison_data_from_csv(config[csv_path_key])
                    all_loaded_data[f"{model_name}_{dataset_type}"] = {
                        'Times': times,
                        'SOC_TRUE': soc_true,
                        'SOC_Pred': soc_pred,
                        'SOE_TRUE': soe_true,
                        'SOE_Pred': soe_pred
                    }
                except FileNotFoundError as e:
                    print(f"   跳过 {model_name} {dataset_type}: {e}")
                except KeyError as e:
                    print(f"   跳过 {model_name} {dataset_type}: {e}")
            else:
                print(f"   跳过 {model_name} {dataset_type}: 未提供有效的CSV路径")


    gc.collect()

    # 定义用户提供的平均RMSE和平均MAE值
    user_provided_avg_rmse = {
        "TabPFN": 1.625,
        "FE-KAN-T": 1.417
    }

    user_provided_avg_mae = {
        "TabPFN": 1.286,
        "FE-KAN-T": 1.166
    }

    # --- 1. 生成时间序列对比图 ---
    print("🎨 Generating time series comparison plots (combined LA92 and UDDS)...")
    
    # 确保每个模型都有独特的SCI风格颜色
    model_colors_map = {
        "FE-KAN-T": '#0072BD',          # 深蓝色
        "No Workload Detector": '#800080', # 紫色
        "Only V-I Features": '#FF7F0E',  # 橙色 (来自原始图例)
        "KAN": '#2CA02C',             # 绿色 (来自原始图例)
        "Transformer": '#D62728',      # 红色 (来自原始图例)
        "Independent SOC/SOE": '#8C564B', # 棕色 (来自原始图例)
    }
    
    # 移除默认颜色迭代器和过滤逻辑, 因为所有模型都将有明确的颜色
    # all_tab10_colors = list(plt.cm.tab10.colors)
    # used_colors = set(model_colors_map.values())
    # filtered_default_colors = [c for c in all_tab10_colors if c not in used_colors]
    
    selected_models_for_ts_plot = list(MODELS_CONFIG.keys()) 

    fig = plt.figure(figsize=(18, 16)) 
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.2) 
    
    fig.suptitle('Prediction and Error Comparison Across Models and Drive Cycles', 
                 fontsize=16, fontweight='bold', y=0.98) 

    axes_grid = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]
    
    subplot_labels = iter([f'({chr(97+i)})' for i in range(8)]) 
    
    datasets_to_plot = ["LA92", "UDDS"]
    metrics_to_plot = [("SOC", "SOC Prediction", "SOC Error (%)", "SOC (%)"), 
                       ("SOE", "SOE Prediction", "SOE Error (%)", "SOE (%)")]

    row_offset = 0
    for dataset_type in datasets_to_plot:
        # 确保 FE-KAN-T 数据已加载 (作为真实值基准)
        if f"FE-KAN-T_{dataset_type}" not in all_loaded_data:
            print(f"Warning: FE-KAN-T {dataset_type} data not found, skipping plots for {dataset_type}.")
            row_offset += 2 
            continue 

        fe_kan_t_data = all_loaded_data[f"FE-KAN-T_{dataset_type}"]
        fe_kan_t_times = fe_kan_t_data['Times']
        actual_data_len = len(fe_kan_t_times)

        for j, (metric_prefix, pred_title, err_ylabel, pred_ylabel) in enumerate(metrics_to_plot):
            current_ax_pred = axes_grid[row_offset + j][0] 
            current_ax_err = axes_grid[row_offset + j][1]  

            # --- 绘制预测图 - Actual Value 遵循指令的黑色 ---
            current_ax_pred.plot(fe_kan_t_times, fe_kan_t_data[f'{metric_prefix}_TRUE'], color='black', linewidth=1.8, label='Actual Value', alpha=0.8)
            
            # 直接从 model_colors_map 获取颜色
            for model_name in selected_models_for_ts_plot:
                model_plot_color = model_colors_map.get(model_name, 'gray') # 如果模型不在映射中, 使用灰色作为默认值

                if f"{model_name}_{dataset_type}" in all_loaded_data:
                    model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                    pred_metric = model_data[f'{metric_prefix}_Pred']
                    current_true_metric = model_data[f'{metric_prefix}_TRUE'] 
                    current_times = model_data['Times']
                    
                    # 预测图平滑窗口设置为 121
                    current_ax_pred.plot(current_times, smooth_data(pred_metric, window_length=121), 
                                         color=model_plot_color, linewidth=1.8, label=model_name) 
                    
                    # 误差图平滑窗口设置为 9
                    current_ax_err.plot(current_times, smooth_data(pred_metric, window_length=9) - smooth_data(current_true_metric, window_length=9), 
                                         color=model_plot_color, linewidth=1.8, alpha=0.8) 
                else:
                    print(f"Warning: {model_name} {dataset_type} data not found or invalid, skipping for {metric_prefix} plot.")
            
            # --- 预测图设置 ---
            current_ax_pred.set_title(f'{next(subplot_labels)} {dataset_type} {pred_title}', fontsize=12, fontweight='normal') 
            current_ax_pred.set_xlabel('Time(s)', fontsize=11) 
            current_ax_pred.set_ylabel(pred_ylabel, fontsize=11) 
            current_ax_pred.legend(loc='upper right', fontsize=10, frameon=False, ncol=2) 
            current_ax_pred.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            current_ax_pred.set_ylim(0, 100) 

            # --- 误差图设置 ---
            current_ax_err.set_title(f'{next(subplot_labels)} {dataset_type} {metric_prefix} Error', fontsize=12, fontweight='normal') 
            current_ax_err.set_xlabel('Time(s)', fontsize=11) 
            current_ax_err.set_ylabel(err_ylabel, fontsize=11) 
            current_ax_err.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=0.8) 
            current_ax_err.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            current_ax_err.set_ylim(-9, 9) # 修改误差图的Y轴范围为-9到9

            # --- 边框样式 --- 
            for spine in current_ax_pred.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')
            for spine in current_ax_err.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')

            # --- 添加局部放大图 ---
            axins = None 
            zoom_start_default = 0
            zoom_length = 800 

            if dataset_type == "LA92":
                zoom_start_default = 4500 # 修改LA92的放大起始位置为4500
                zoom_length = 1000 
            elif dataset_type == "UDDS":
                zoom_start_default = 5000 # 修改UDDS的放大起始位置为5000
                zoom_length = 800

            zoom_start = zoom_start_default
            zoom_end = min(zoom_start + zoom_length, actual_data_len)

            if zoom_end <= zoom_start:
                print(f"Warning: Cannot create {metric_prefix} zoom for {dataset_type} due to insufficient data length. Skipping inset.")
            else:
                axins = current_ax_pred.inset_axes([0.08, 0.08, 0.4, 0.4], transform=current_ax_pred.transAxes) 
                y_min_zoom = 100
                y_max_zoom = 0

                # 真实值
                true_zoom_segment = fe_kan_t_data[f'{metric_prefix}_TRUE'][(fe_kan_t_times >= zoom_start) & (fe_kan_t_times <= zoom_end)]
                if len(true_zoom_segment) > 0:
                    y_min_zoom = min(y_min_zoom, np.min(true_zoom_segment))
                    y_max_zoom = max(y_max_zoom, np.max(true_zoom_segment))

                # 直接从 model_colors_map 获取颜色
                for model_name in selected_models_for_ts_plot:
                    model_plot_color = model_colors_map.get(model_name, 'gray') 

                    if f"{model_name}_{dataset_type}" in all_loaded_data:
                        model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                        pred_metric = model_data[f'{metric_prefix}_Pred']
                        current_times = model_data['Times']

                        pred_metric_zoom = pred_metric[(current_times >= zoom_start) & (current_times <= zoom_end)]
                        if len(pred_metric_zoom) > 0:
                            y_min_zoom = min(y_min_zoom, np.min(smooth_data(pred_metric_zoom, window_length=121))) # 预测图窗口保持121
                            y_max_zoom = max(y_max_zoom, np.max(smooth_data(pred_metric_zoom, window_length=121))) # 预测图窗口保持121

                y_range_zoom = y_max_zoom - y_min_zoom
                if y_range_zoom == 0: y_range_zoom = 1

                # 绘制 inset 图
                axins.plot(fe_kan_t_times[(fe_kan_t_times >= zoom_start) & (fe_kan_t_times <= zoom_end)], 
                           true_zoom_segment, color='black', linewidth=1.8, label='Actual Value', alpha=0.8)
                
                # 确保 inset 中的颜色与主图一致
                for model_name in selected_models_for_ts_plot:
                    model_plot_color = model_colors_map.get(model_name, 'gray') 

                    if f"{model_name}_{dataset_type}" in all_loaded_data:
                        model_data = all_loaded_data[f"{model_name}_{dataset_type}"]
                        pred_metric = model_data[f'{metric_prefix}_Pred']
                        current_times = model_data['Times']

                        axins.plot(current_times[(current_times >= zoom_start) & (current_times <= zoom_end)], 
                                   smooth_data(pred_metric[(current_times >= zoom_start) & (current_times <= zoom_end)], window_length=121), 
                                   color=model_plot_color, linewidth=1.8, label=model_name) # 预测图窗口保持121
                
                axins.set_xlim(zoom_start, zoom_end)
                axins.set_ylim(y_min_zoom - 0.05 * y_range_zoom, y_max_zoom + 0.05 * y_range_zoom)
                axins.set_xticks([])
                axins.set_yticks([])
                axins.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                for spine in axins.spines.values(): 
                    spine.set_linewidth(1.0)
                    spine.set_color('black')
                current_ax_pred.indicate_inset_zoom(axins, edgecolor='darkred', linewidth=1.0)
        
        row_offset += 2 

    fig.text(0.5, 0.03, 'Time(s)', ha='center', va='center', fontsize=11)

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.96]) 
    plt.savefig(os.path.join(args.output_dir, f'Figure_Time_Series_Comparison_All_Models.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='png', pad_inches=0.1)
    plt.close(fig) 
    print(f"   Saved Figure_Time_Series_Comparison_All_Models.png")

    # --- 2. 生成RMSE和MAE对比柱状图 ---
    print("📊 Generating RMSE and MAE comparison bar chart...")
    
    # 用户提供的所有模型及其分工况的RMSE和MAE数据
    model_names_ordered = ["TabPFN", "FE-KAN-T"] # 移除 N-BEATS 和 CNN-GRU
    
    # 注意：这里的数据顺序必须与model_names_ordered一致
    # 调整数据以匹配新的 model_names_ordered
    la92_rmses = [2.193, 1.547]
    udds_rmses = [1.0555, 1.0195]
    la92_maes = [1.7815, 1.334]
    udds_maes = [0.7915, 0.774]

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
    # 对模型按计算出的平均RMSE排序, 并将FE-KAN-T放在最前面
    other_models_for_sort = [m for m in model_names_ordered if m != "FE-KAN-T"]
    sorted_other_models = sorted(other_models_for_sort, key=lambda x: results_df_for_plot[results_df_for_plot['Model']==x]['Average_RMSE'].iloc[0])
    sorted_models = ["FE-KAN-T"] + sorted_other_models

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
