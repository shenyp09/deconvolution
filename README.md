# 能谱生成与反卷积处理

这个项目实现了能谱数据的生成和反卷积处理功能。

## 功能

1. **数据产生模块**：
   - 生成长度为8192的序列
   - 生成连续分布的原始能谱，形状为随机光滑连续结构
   - 添加宽度为1道的高强度特征峰
   - 将原始能谱与高斯函数卷积（sigma=rev*sqrt(E)）
   - 为实际能谱增加统计误差（误差值为A*SQRT(Amp_N)）
   - 使用HTML可视化结果

2. **反卷积模块**：
   - 支持多种先进反卷积算法
   - 对生成的实际能谱进行反卷积
   - 与原始能谱进行对比
   - 使用HTML可视化结果

## 支持的反卷积算法

1. **混合反卷积（hybrid）**：结合自适应反卷积和Richardson-Lucy算法
2. **稀疏反卷积（sparse）**：基于ISTA算法，特别适合恢复尖锐峰
3. **盲反卷积（blind）**：同时估计点扩散函数和原始信号
4. **ADMM反卷积（admm）**：基于交替方向乘子法的反卷积
5. **贝叶斯反卷积（bayesian）**：利用先验知识改善反卷积结果
6. **自适应反卷积（adaptive）**：根据信号局部特性自适应调整参数

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
- `--use_wiener`：启用Wiener滤波（标志参数）
- `--snr`：Wiener滤波的信噪比参数（默认：100.0）
- `--no_peak_enhancement`：禁用峰增强模式（标志参数）
- `--algorithm`：选择反卷积算法，可选值：hybrid, sparse, blind, admm, bayesian, adaptive（默认：hybrid）
- `--peak_intensity`：特征峰强度参数（默认：30000.0）

例如：

```bash
# 使用稀疏反卷积算法处理2个能谱
python energy_spectrum.py --num_spectra 2 --algorithm sparse

# 使用贝叶斯反卷积算法并启用Wiener滤波
python energy_spectrum.py --algorithm bayesian --use_wiener --snr 50

# 使用ADMM算法处理高分辨率的能谱
python energy_spectrum.py --algorithm admm --rev 0.005

# 设置更高的特征峰强度
python energy_spectrum.py --peak_intensity 50000

# 低强度特征峰与盲反卷积算法组合
python energy_spectrum.py --peak_intensity 10000 --algorithm blind
```

## 输出

程序会生成两个HTML文件：
- `energy_spectra.html`：包含生成的原始能谱和卷积后带噪声的能谱
- `deconvolution_results.html`：包含原始能谱、卷积后带噪声的能谱以及反卷积结果的对比

## 算法性能比较

不同算法在处理尖锐特征峰时性能各异：

- **稀疏反卷积**：对尖锐峰恢复效果最好，但噪声敏感性较高
- **贝叶斯反卷积**：平衡了峰恢复和平滑性，适合信噪比较低的情况
- **ADMM反卷积**：提供了很好的平衡，但计算速度较慢
- **盲反卷积**：当点扩散函数不确定时表现最佳
- **混合算法**：综合表现最稳定，是默认选择

## 特征峰强度说明

- 特征峰强度参数控制生成的尖锐特征峰的高度
- 较高的强度值（如50000）可以生成明显的特征峰，更容易被反卷积识别
- 较低的强度值（如10000）可以测试算法对弱特征的恢复能力
- 默认值30000适合大多数测试场景 