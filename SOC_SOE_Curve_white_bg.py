# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from datetime import datetime
import os
import platform

# 根据操作系统设置合适的中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
file_path = r"C:\Users\黎枭\Desktop\10triseLA92.csv"
try:
    data = pd.read_csv(file_path)
    print(f"成功读取数据，共{len(data)}行")
    
    # 确保时间从0开始
    if 'Time' in data.columns:
        min_time = data['Time'].min()
        data['Time'] = data['Time'] - min_time  # 调整时间使其从0开始
        
except FileNotFoundError:
    print(f"文件不存在: {file_path}")
    print("创建模拟数据用于演示...")
    # 创建模拟数据用于演示
    np.random.seed(42)
    n_points = 100
    time = np.linspace(0, 14000, n_points)  # 使用与示例图像相似的时间范围，从0开始
    soc = 1 - time/14000 + np.random.normal(0, 0.01, n_points)  # 线性下降趋势
    soc = np.clip(soc, 0, 1)
    soe = 1 - time/14000*1.05 + np.random.normal(0, 0.01, n_points)  # 略微快一点的下降
    soe = np.clip(soe, 0, 1)
    
    data = pd.DataFrame({
        'Time': time,
        'SOC': soc,
        'SOE': soe
    })

# 假设数据中有SOC和SOE列，如果没有，需要根据实际数据计算
if 'SOC' not in data.columns or 'SOE' not in data.columns:
    print("请确认数据中是否包含SOC和SOE列，或提供计算方法")
    # 如果需要计算SOC和SOE，请在此添加计算代码

# 假设数据中有时间列，如果是时间戳格式，需要转换为datetime
if 'Time' in data.columns:
    time_col = 'Time'
elif '时间' in data.columns:
    time_col = '时间'
else:
    # 尝试找到可能的时间列
    time_cols = [col for col in data.columns if 'time' in col.lower() or '时间' in col]
    if time_cols:
        time_col = time_cols[0]
    else:
        # 如果没有时间列，创建一个序列作为时间
        data['Time'] = range(len(data))
        time_col = 'Time'

# 创建图表 - 使用白色背景
fig, ax = plt.subplots(figsize=(8, 3), dpi=300, facecolor='white')  # 设置画布背景为白色

# 设置白色背景
ax.set_facecolor('white')  # 白色背景
fig.patch.set_facecolor('white')  # 图形背景也设为白色

# 绘制SOC和SOE曲线，使用红色和蓝色
if 'SOC' in data.columns:
    ax.plot(data[time_col], data['SOC'], '-', color='red', linewidth=1.5, 
            label='SOC')
if 'SOE' in data.columns:
    ax.plot(data[time_col], data['SOE'], '-', color='#6495ED', linewidth=1.5,  # 浅蓝色
            label='SOE')

# 设置坐标轴范围和标签
ax.set_ylim(0, 1.0)  # 从0到1.0
ax.set_xlim(0, data[time_col].max() * 1.02)  # 确保从0开始，给右侧留一点空间

# 添加双Y轴标签
ax.set_ylabel('SOC/SOE', fontsize=12, fontweight='bold')
ax.yaxis.set_label_coords(-0.05, 0.5)  # 调整Y轴标签位置

# 添加右侧Y轴，与左侧相同
ax2 = ax.twinx()
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('SOC/SOE', fontsize=12, fontweight='bold')

# 设置X轴标签
ax.set_xlabel('Time(s)', fontsize=12, fontweight='bold')

# 美化坐标轴
ax.tick_params(axis='both', which='major', labelsize=10, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=8, direction='in')
ax2.tick_params(axis='both', which='major', labelsize=10, direction='in')

# 设置主网格线和次网格线 - 使用更浅的颜色
ax.grid(True, linestyle='-', alpha=0.15, color='gray')

# 添加图例，放在右上角
ax.legend(loc='upper right', fontsize=10, frameon=False)

# 设置坐标轴的线宽
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)

# 保存图表 - 确保透明度设置
plt.tight_layout()
plt.savefig('SOC_SOE_Curve_white_bg.png', dpi=600, bbox_inches='tight', transparent=False)
plt.savefig('SOC_SOE_Curve_white_bg.pdf', format='pdf', dpi=600, bbox_inches='tight', transparent=False)
plt.savefig('SOC_SOE_Curve_white_bg.tif', format='tiff', dpi=600, bbox_inches='tight', transparent=False)

# 也保存透明背景版本
plt.savefig('SOC_SOE_Curve_transparent.png', dpi=600, bbox_inches='tight', transparent=True)

print('图像已生成：白色背景和透明背景版本')

# 显示图表
plt.show() 