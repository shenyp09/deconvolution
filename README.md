# 能谱生成与反卷积处理

这个项目实现了能谱数据的生成和反卷积处理功能。

## 功能

1. **数据产生模块**：
   - 生成长度为8192的序列
   - 生成连续分布的原始能谱，形状为随机光滑连续结构
   - 将原始能谱与高斯函数卷积（sigma=rev*sqrt(E)）
   - 为实际能谱增加统计误差（误差值为A*SQRT(Amp_N)）
   - 使用HTML可视化结果

2. **反卷积模块**：
   - 使用Gold迭代方法进行反卷积
   - 对生成的实际能谱进行反卷积
   - 与原始能谱进行对比
   - 使用HTML可视化结果

## 依赖

- numpy
- scipy
- matplotlib
- plotly

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：

```bash
python energy_spectrum.py
```

### 参数选项

- `--num_spectra`：生成的能谱数量（默认：10）
- `--rev`：分辨率参数，影响高斯卷积的宽度（默认：0.01）
- `--error_amplitude`：统计误差强度参数（默认：1.0）

例如：

```bash
python energy_spectrum.py --num_spectra 5 --rev 0.02 --error_amplitude 0.5
```

## 输出

程序会生成两个HTML文件：
- `energy_spectra.html`：包含生成的原始能谱和卷积后带噪声的能谱
- `deconvolution_results.html`：包含原始能谱、卷积后带噪声的能谱以及反卷积结果的对比 