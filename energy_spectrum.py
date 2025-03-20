import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import time

class EnergySpectrumGenerator:
    def __init__(self, length=8192, num_spectra=10, rev=1, error_amplitude=1.0, peak_intensity=30000):
        self.length = length
        self.num_spectra = num_spectra
        self.rev = rev  # 高斯卷积的分辨率参数
        self.error_amplitude = error_amplitude  # 统计误差强度参数
        self.peak_intensity = peak_intensity  # 特征峰强度参数
        self.x = np.linspace(1, length, length)  # 能量通道
        self.original_spectra = []
        self.convolved_spectra = []
        self.noisy_spectra = []
        
    def generate_smooth_spectrum(self):
        """生成随机光滑连续结构的能谱"""
        # 创建一些随机峰
        num_peaks = np.random.randint(3, 8)
        spectrum = np.zeros(self.length)
        
        for _ in range(num_peaks):
            # 随机峰位置
            center = np.random.randint(500, self.length - 500)
            # 随机峰宽度
            width = np.random.randint(100, 500)
            # 随机峰高度
            height = np.random.uniform(50, 200)
            
            # 生成高斯峰
            peak = height * np.exp(-0.5 * ((self.x - center) / width) ** 2)
            spectrum += peak
        
        # 添加随机的背景
        background = np.random.uniform(5, 20) * np.exp(-self.x / np.random.uniform(2000, 5000))
        spectrum += background
        
        # 平滑整个谱
        spectrum = gaussian_filter1d(spectrum, sigma=50)
        
        # 添加特征峰
        # 1. 添加几个窄的高斯特征峰
        num_feature_peaks = np.random.randint(2, 5)  # 添加2-4个特征峰
        
        for _ in range(num_feature_peaks):
            # 特征峰位置 - 在能谱的不同区域
            region = np.random.choice(['低能', '中能', '高能'])
            if region == '低能':
                center = np.random.randint(500, 2000)
            elif region == '中能':
                center = np.random.randint(2000, 5000)
            else:  # 高能
                center = np.random.randint(5000, 7500)
                
            # 特征峰宽度设为1道
            width = 1
            # 使用设置的特征峰强度，并添加一些随机变化
            height = self.peak_intensity * np.random.uniform(0.8, 1.2)
            
            # 生成高斯特征峰
            feature_peak = np.zeros(self.length)
            feature_peak[center] = height
            spectrum += feature_peak
            
        
        
        return spectrum
        
    def apply_convolution(self, spectrum):
        """应用能量依赖的高斯卷积"""
        convolved = np.zeros_like(spectrum)
        
        for i in range(self.length):
            # 能量依赖的分辨率
            energy = self.x[i]
            sigma = self.rev * np.sqrt(energy)
            
            # 创建高斯核
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = min(kernel_size, 201)  # 限制核大小
            
            if kernel_size > 2:
                x_kernel = np.linspace(-3*sigma, 3*sigma, kernel_size)
                kernel = np.exp(-0.5 * (x_kernel / sigma) ** 2)
                kernel = kernel / np.sum(kernel)
                
                # 应用卷积
                low = max(0, i - kernel_size // 2)
                high = min(self.length, i + kernel_size // 2 + 1)
                k_low = max(0, kernel_size // 2 - i)
                k_high = kernel_size - max(0, i + kernel_size // 2 + 1 - self.length)
                
                if high > low and k_high > k_low:
                    convolved[i] = np.sum(spectrum[low:high] * kernel[k_low:k_high])
            else:
                convolved[i] = spectrum[i]
                
        return convolved
    
    def add_statistical_noise(self, spectrum):
        """添加统计误差"""
        # 噪声大小与信号幅度的平方根成正比
        noise = np.random.normal(0, 1, self.length) * self.error_amplitude * np.sqrt(np.maximum(spectrum, 0))
        return spectrum + noise
    
    def generate_data(self):
        """生成多个能谱数据"""
        self.original_spectra = []
        self.convolved_spectra = []
        self.noisy_spectra = []
        
        print("正在生成能谱数据...")
        start_time = time.time()
        
        for i in range(self.num_spectra):
            # 生成原始能谱
            original = self.generate_smooth_spectrum()
            self.original_spectra.append(original)
            
            # 应用卷积
            convolved = self.apply_convolution(original)
            self.convolved_spectra.append(convolved)
            
            # 添加统计误差
            noisy = self.add_statistical_noise(convolved)
            self.noisy_spectra.append(noisy)
            
            print(f"已生成 {i+1}/{self.num_spectra} 个能谱")
        
        print(f"数据生成完成，耗时 {time.time() - start_time:.2f} 秒")
        
        return self.original_spectra, self.convolved_spectra, self.noisy_spectra
    
    def visualize_html(self, output_file="energy_spectra.html"):
        """使用Plotly生成HTML可视化"""
        fig = make_subplots(rows=self.num_spectra, cols=1, 
                           subplot_titles=[f"能谱 {i+1}" for i in range(self.num_spectra)],
                           vertical_spacing=0.05)
        
        for i in range(self.num_spectra):
            fig.add_trace(
                go.Scatter(x=self.x, y=self.original_spectra[i], mode='lines', name=f'原始能谱 {i+1}', 
                          line=dict(color='blue')),
                row=i+1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=self.x, y=self.noisy_spectra[i], mode='lines', name=f'卷积后带噪声能谱 {i+1}', 
                          line=dict(color='red')),
                row=i+1, col=1
            )
        
        fig.update_layout(height=500*self.num_spectra, width=1000, 
                         title="能谱数据集可视化",
                         showlegend=False)
        
        fig.write_html(output_file)
        print(f"可视化结果已保存至 {output_file}")
        
        return fig


class DeconvolutionProcessor:
    def __init__(self, rev=0.01, use_wiener=False, snr=100.0, peak_enhancement=True, algorithm='hybrid'):
        self.rev = rev  # 卷积参数
        self.use_wiener = use_wiener  # 是否使用Wiener滤波
        self.snr = snr  # Wiener滤波的信噪比参数
        self.peak_enhancement = peak_enhancement  # 是否启用峰增强模式
        self.algorithm = algorithm  # 使用的反卷积算法
        
    def find_potential_peaks(self, spectrum, threshold_factor=3.0, min_width=1, max_width=10):
        """寻找可能的尖锐特征峰位置"""
        length = len(spectrum)
        # 计算局部均值和标准差
        window_size = 101
        local_mean = np.zeros_like(spectrum)
        local_std = np.zeros_like(spectrum)
        
        for i in range(length):
            start = max(0, i - window_size // 2)
            end = min(length, i + window_size // 2 + 1)
            local_data = spectrum[start:end]
            local_mean[i] = np.mean(local_data)
            local_std[i] = np.std(local_data)
        
        # 确定峰的位置（高于局部平均+n*标准差）
        potential_peaks = []
        i = 0
        while i < length:
            if spectrum[i] > local_mean[i] + threshold_factor * local_std[i]:
                # 找到峰的开始
                peak_start = i
                
                # 找到峰的最高点
                peak_max = peak_start
                while i < length and spectrum[i] > local_mean[i] + threshold_factor * local_std[i]:
                    if spectrum[i] > spectrum[peak_max]:
                        peak_max = i
                    i += 1
                
                # 找到峰的结束
                peak_end = i - 1
                
                # 计算峰的宽度
                peak_width = peak_end - peak_start + 1
                
                # 只关注符合宽度条件的峰
                if min_width <= peak_width <= max_width:
                    potential_peaks.append((peak_max, peak_width, spectrum[peak_max]))
            else:
                i += 1
        
        # 按峰高度排序
        potential_peaks.sort(key=lambda x: x[2], reverse=True)
        
        return potential_peaks
        
    def create_response_matrix(self, length):
        """创建完整的响应矩阵，用于精确反卷积"""
        response_matrix = np.zeros((length, length))
        x = np.arange(1, length+1)
        
        for i in range(length):
            energy = x[i]
            sigma = self.rev * np.sqrt(energy)
            
            # 对每个能量点创建高斯响应
            for j in range(length):
                delta_e = x[j] - energy
                response_matrix[i, j] = np.exp(-0.5 * (delta_e / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        # 归一化每行
        row_sums = response_matrix.sum(axis=1, keepdims=True)
        response_matrix = response_matrix / np.where(row_sums > 0, row_sums, 1)
        
        return response_matrix
    
    def create_sparse_response_matrix(self, length):
        """创建稀疏响应矩阵，只计算有意义的元素"""
        response_matrix = np.zeros((length, length))
        x = np.arange(1, length+1)
        
        for i in range(length):
            energy = x[i]
            sigma = self.rev * np.sqrt(energy)
            
            # 只计算3个sigma范围内的响应
            window = int(6 * sigma)
            start_j = max(0, i - window)
            end_j = min(length, i + window + 1)
            
            for j in range(start_j, end_j):
                delta_e = x[j] - energy
                response_matrix[i, j] = np.exp(-0.5 * (delta_e / sigma) ** 2)
        
        # 归一化每行
        row_sums = response_matrix.sum(axis=1, keepdims=True)
        response_matrix = response_matrix / np.where(row_sums > 0, row_sums, 1)
        
        return response_matrix
        
    def gold_deconvolution(self, spectrum, iterations=200):
        """使用Gold迭代反卷积算法，无降采样"""
        length = len(spectrum)
        
        # 对于尖锐特征的反卷积，使用稀疏响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 初始估计（平滑的输入谱）
        estimate = gaussian_filter1d(spectrum, sigma=2)
        estimate = np.maximum(estimate, 1e-10)  # 避免零值
        
        # Gold迭代
        for iter in range(iterations):
            # 计算当前估计的卷积结果
            forward = np.dot(response_matrix, estimate)
            forward = np.maximum(forward, 1e-10)  # 避免零值
            
            # 计算比率
            ratio = spectrum / forward
            
            # 应用比率更新估计
            correction = np.dot(response_matrix.T, ratio)
            estimate = estimate * correction
            
            # 避免负值或过小的值
            estimate = np.maximum(estimate, 1e-10)
            
            # 应用正则化以减少噪声放大
            if iter % 10 == 0 and iter > 0:
                estimate = gaussian_filter1d(estimate, sigma=0.5)
            
            if iter % 20 == 0:
                print(f"  反卷积迭代: {iter}/{iterations}")
        
        return estimate
    
    def wiener_deconvolution(self, spectrum):
        """使用Wiener滤波进行反卷积"""
        length = len(spectrum)
        x = np.arange(1, length+1)
        
        # 创建频域响应函数
        freqs = np.fft.fftfreq(length)
        response = np.zeros(length, dtype=complex)
        
        # 使用设置的信噪比
        snr = self.snr
        
        # 在频域中创建卷积核
        for i in range(length):
            energy = x[i]
            sigma = self.rev * np.sqrt(energy)
            kernel = np.exp(-0.5 * (x - energy)**2 / sigma**2)
            kernel = kernel / np.sum(kernel)
            
            # 对每个能量点计算频域响应
            kernel_fft = np.fft.fft(np.roll(kernel, length//2 - i))
            response += kernel_fft / length
        
        # 归一化响应
        response = response / np.abs(response).max()
        
        # 应用Wiener滤波
        spectrum_fft = np.fft.fft(spectrum)
        wiener_filter = np.conj(response) / (np.abs(response)**2 + 1/snr)
        result_fft = spectrum_fft * wiener_filter
        result = np.real(np.fft.ifft(result_fft))
        
        # 确保非负
        result = np.maximum(result, 0)
        
        return result
    
    def richardson_lucy_deconvolution(self, spectrum, iterations=50):
        """使用Richardson-Lucy反卷积算法"""
        length = len(spectrum)
        x = np.arange(1, length+1)
        
        # 创建响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 初始估计
        estimate = np.copy(spectrum)
        estimate = np.maximum(estimate, 1e-10)  # 避免零值
        
        # Richardson-Lucy迭代
        for iter in range(iterations):
            # 计算当前估计的卷积结果
            forward = np.dot(response_matrix, estimate)
            forward = np.maximum(forward, 1e-10)  # 避免零值
            
            # 计算比率并更新估计
            ratio = spectrum / forward
            estimate = estimate * np.dot(response_matrix.T, ratio)
            
            # 确保非负
            estimate = np.maximum(estimate, 0)
            
            # 应用适度的正则化
            if iter % 10 == 0 and iter > 0:
                estimate = gaussian_filter1d(estimate, sigma=0.5)
            
            if iter % 10 == 0:
                print(f"  RL反卷积迭代: {iter}/{iterations}")
        
        return estimate
        
    def optimize_deconvolution(self, spectrum, iterations=50, downsample_factor=4):
        """使用降采样优化的Gold反卷积算法"""
        # 对输入谱进行降采样以加速计算
        original_length = len(spectrum)
        reduced_length = original_length // downsample_factor
        
        # 降采样
        downsampled_spectrum = np.zeros(reduced_length)
        for i in range(reduced_length):
            downsampled_spectrum[i] = np.mean(spectrum[i*downsample_factor:(i+1)*downsample_factor])
        
        # 在降采样的空间中创建响应矩阵
        response_matrix = np.zeros((reduced_length, reduced_length))
        downsampled_x = np.linspace(1, original_length, reduced_length)
        
        # 优化的响应矩阵构建 - 只计算主对角线附近的元素
        for i in range(reduced_length):
            energy = downsampled_x[i]
            sigma = self.rev * np.sqrt(energy) / downsample_factor
            
            # 确定影响范围（只考虑3个sigma范围内的点）
            window = int(3 * sigma * downsample_factor)
            window = max(10, window)  # 确保至少有一些点
            
            start_j = max(0, i - window)
            end_j = min(reduced_length, i + window + 1)
            
            for j in range(start_j, end_j):
                delta_e = downsampled_x[j] - energy
                response_matrix[i, j] = np.exp(-0.5 * (delta_e / (sigma * downsample_factor)) ** 2)
        
        # 归一化每行
        row_sums = response_matrix.sum(axis=1, keepdims=True)
        response_matrix = response_matrix / np.where(row_sums > 0, row_sums, 1)
        
        # 初始估计（平滑的输入谱）
        estimate = gaussian_filter1d(downsampled_spectrum, sigma=2)
        estimate = np.maximum(estimate, 1e-10)  # 避免零值
        
        # Gold迭代
        for iter in range(iterations):
            # 计算当前估计的卷积结果
            forward = np.dot(response_matrix, estimate)
            forward = np.maximum(forward, 1e-10)  # 避免零值
            
            # 计算比率
            ratio = downsampled_spectrum / forward
            
            # 应用比率更新估计
            correction = np.dot(response_matrix.T, ratio)
            estimate = estimate * correction
            
            # 避免负值或过小的值
            estimate = np.maximum(estimate, 1e-10)
            
            if iter % 10 == 0:
                print(f"  反卷积迭代: {iter}/{iterations}")
        
        # 上采样回原始大小
        upsampled_estimate = np.zeros(original_length)
        for i in range(reduced_length):
            upsampled_estimate[i*downsample_factor:(i+1)*downsample_factor] = estimate[i]
        
        # 最终平滑处理
        upsampled_estimate = gaussian_filter1d(upsampled_estimate, sigma=1)
        
        return upsampled_estimate
    
    def hybrid_deconvolution(self, spectrum):
        """混合反卷积策略，针对尖锐特征峰优化"""
        # 首先使用Gold算法获取初步结果
        print("  第一阶段: Gold算法反卷积...")
        gold_result = self.gold_deconvolution(spectrum, iterations=100)
        
        # 如果启用Wiener滤波，添加Wiener滤波步骤
        if self.use_wiener:
            print("  Wiener滤波处理...")
            wiener_result = self.wiener_deconvolution(spectrum)
            # 将Wiener结果与Gold结果结合
            combined_result = (gold_result + wiener_result) / 2
            # 使用加权平均结果作为下一阶段的初始估计
            initial_estimate = combined_result
        else:
            initial_estimate = gold_result
        
        # 然后使用Richardson-Lucy算法改善细节
        print("  第二阶段: Richardson-Lucy算法精细反卷积...")
        # 使用前面的结果作为初始估计
        length = len(spectrum)
        
        # 创建响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 设置初始估计
        estimate = initial_estimate.copy()
        estimate = np.maximum(estimate, 1e-10)
        
        # Richardson-Lucy精细迭代
        for iter in range(30):
            # 计算当前估计的卷积结果
            forward = np.dot(response_matrix, estimate)
            forward = np.maximum(forward, 1e-10)
            
            # 计算比率并更新估计
            ratio = spectrum / forward
            estimate = estimate * np.dot(response_matrix.T, ratio)
            
            # 确保非负
            estimate = np.maximum(estimate, 0)
            
            if iter % 10 == 0:
                print(f"  精细反卷积迭代: {iter}/30")
        
        return estimate
    
    def sparse_deconvolution(self, spectrum, iterations=50, lambda_reg=0.01):
        """使用稀疏反卷积算法，特别优化尖锐特征峰的恢复"""
        length = len(spectrum)
        
        # 创建响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 初始估计
        estimate = np.copy(spectrum)
        estimate = np.maximum(estimate, 1e-10)
        
        print("  开始稀疏反卷积(ISTA)...")
        # ISTA迭代（迭代收缩阈值算法）
        for iter in range(iterations):
            # 计算梯度
            forward = np.dot(response_matrix, estimate)
            residual = spectrum - forward
            gradient = np.dot(response_matrix.T, residual)
            
            # 更新估计
            step_size = 0.05  # 步长
            temp = estimate + step_size * gradient
            
            # 软阈值（促进稀疏性）
            threshold = lambda_reg * step_size
            estimate = np.sign(temp) * np.maximum(np.abs(temp) - threshold, 0)
            
            # 处理负值
            estimate = np.maximum(estimate, 0)
            
            if iter % 10 == 0:
                print(f"  稀疏反卷积迭代: {iter}/{iterations}")
        
        # 针对可能的峰值位置进行增强
        if self.peak_enhancement:
            print("  应用峰值增强...")
            peak_candidates = self.find_potential_peaks(spectrum)
            for peak_pos, _, _ in peak_candidates:
                # 在峰值周围应用更激进的增强
                window = 2
                start = max(0, peak_pos - window)
                end = min(length, peak_pos + window + 1)
                
                if start < end and peak_pos < length:
                    # 增强峰值
                    peak_value = estimate[peak_pos] * 1.5
                    estimate[peak_pos] = peak_value
        
        return estimate
    
    def _create_psf_matrix(self, psf, length):
        """根据PSF创建卷积矩阵"""
        psf_len = len(psf)
        psf_center = psf_len // 2
        psf_matrix = np.zeros((length, length))
        
        for i in range(length):
            for j in range(max(0, i-psf_center), min(length, i+psf_len-psf_center)):
                if 0 <= j-i+psf_center < psf_len:
                    psf_matrix[i, j] = psf[j-i+psf_center]
        
        return psf_matrix
    
    def _update_psf(self, spectrum, estimate, psf_len=21):
        """更新点扩散函数的估计"""
        # 创建Toeplitz矩阵
        length = len(spectrum)
        A = np.zeros((length, psf_len))
        
        for i in range(length):
            for j in range(psf_len):
                idx = i - (j - psf_len//2)
                if 0 <= idx < length:
                    A[i, j] = estimate[idx]
        
        # 使用非负最小二乘解决Ax = b
        from scipy.optimize import nnls
        psf_new, _ = nnls(A, spectrum)
        
        # 归一化
        psf_sum = np.sum(psf_new)
        if psf_sum > 0:
            psf_new = psf_new / psf_sum
        
        return psf_new
    
    def blind_deconvolution(self, spectrum, iterations=20):
        """盲反卷积方法，同时估计点扩散函数和原始信号"""
        length = len(spectrum)
        
        # 初始估计
        estimate = np.copy(spectrum)
        
        # 初始PSF估计（点扩散函数）
        psf_len = 21
        psf_estimate = np.zeros(psf_len)
        psf_estimate[psf_len//2] = 1.0  # 初始为单位脉冲
        
        print("  开始盲反卷积...")
        for iter in range(iterations):
            # 1. 使用当前PSF估计更新信号估计
            psf_matrix = self._create_psf_matrix(psf_estimate, length)
            
            # 使用Wiener滤波进行反卷积
            spectrum_fft = np.fft.fft(spectrum)
            psf_fft = np.fft.fft(np.pad(psf_estimate, (0, length - psf_len)))
            snr = self.snr
            
            # 应用Wiener滤波
            wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + 1/snr)
            result_fft = spectrum_fft * wiener_filter
            estimate = np.real(np.fft.ifft(result_fft))
            
            # 确保非负
            estimate = np.maximum(estimate, 0)
            
            # 2. 使用当前信号估计更新PSF
            if iter < iterations - 1:  # 最后一次迭代不更新PSF
                psf_new = self._update_psf(spectrum, estimate, psf_len)
                psf_estimate = psf_new
            
            if iter % 5 == 0:
                print(f"  盲反卷积迭代: {iter}/{iterations}")
        
        # 峰值增强
        if self.peak_enhancement:
            print("  应用峰值增强...")
            peak_candidates = self.find_potential_peaks(spectrum)
            for peak_pos, _, _ in peak_candidates:
                # 在峰值周围应用增强
                window = 2
                start = max(0, peak_pos - window)
                end = min(length, peak_pos + window + 1)
                
                if start < end and peak_pos < length:
                    # 增强峰值
                    peak_value = estimate[peak_pos] * 1.3
                    estimate[peak_pos] = peak_value
        
        return estimate
    
    def admm_deconvolution(self, spectrum, iterations=50, rho=1.0):
        """使用ADMM（交替方向乘子法）进行反卷积"""
        length = len(spectrum)
        
        # 创建响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 初始化变量
        x = np.copy(spectrum)  # 主变量（估计的原始谱）
        z = np.copy(x)         # 辅助变量
        u = np.zeros_like(x)   # 拉格朗日乘子
        
        print("  开始ADMM反卷积...")
        for iter in range(iterations):
            # 更新x（使用伪逆解决线性系统）
            RTR = response_matrix.T @ response_matrix
            I = np.eye(length)
            lhs = RTR + rho * I
            rhs = response_matrix.T @ spectrum + rho * (z - u)
            
            from scipy.sparse.linalg import cg
            x, _ = cg(lhs, rhs, x0=x, maxiter=10)
            
            # 更新z（促进稀疏性的软阈值操作）
            z_old = z.copy()
            z = np.maximum(x + u - 0.1/rho, 0)  # L1正则化的软阈值
            
            # 更新拉格朗日乘子
            u = u + (x - z)
            
            # 检查收敛性
            x_change = np.linalg.norm(x - z)
            z_change = np.linalg.norm(z - z_old)
            
            if iter % 10 == 0:
                print(f"  ADMM迭代: {iter}/{iterations}, 残差: {x_change:.6f}")
                
            if x_change < 1e-4 and z_change < 1e-4:
                print(f"  ADMM提前收敛于迭代 {iter}")
                break
        
        # 非负约束
        x = np.maximum(x, 0)
        
        # 峰值增强
        if self.peak_enhancement:
            print("  应用峰值增强...")
            peak_candidates = self.find_potential_peaks(spectrum)
            for peak_pos, _, _ in peak_candidates:
                window = 2
                start = max(0, peak_pos - window)
                end = min(length, peak_pos + window + 1)
                
                if start < end and peak_pos < length:
                    peak_value = x[peak_pos] * 1.4
                    x[peak_pos] = peak_value
        
        return x
    
    def bayesian_deconvolution(self, spectrum, iterations=50):
        """贝叶斯反卷积方法，使用先验知识改善结果"""
        length = len(spectrum)
        
        # 创建响应矩阵
        response_matrix = self.create_sparse_response_matrix(length)
        
        # 初始估计
        estimate = gaussian_filter1d(spectrum, sigma=1)
        estimate = np.maximum(estimate, 1e-10)
        
        # 寻找可能的峰位置作为先验知识
        potential_peaks = self.find_potential_peaks(spectrum)
        
        # 创建先验分布（在峰位置周围有较高概率）
        prior = np.ones(length) * 0.1  # 基础概率
        for peak_pos, peak_width, _ in potential_peaks:
            # 在峰位置增加先验概率
            window = max(3, peak_width)
            start = max(0, peak_pos - window)
            end = min(length, peak_pos + window + 1)
            prior[start:end] = 1.0
            if 0 <= peak_pos < length:
                prior[peak_pos] = 2.0  # 峰位置处最高概率
        
        print("  开始贝叶斯反卷积...")
        for iter in range(iterations):
            # 1. 计算当前估计的前向投影
            forward = np.dot(response_matrix, estimate)
            forward = np.maximum(forward, 1e-10)
            
            # 2. 计算估计与实际数据的比率
            ratio = spectrum / forward
            
            # 3. 计算更新系数
            update = np.dot(response_matrix.T, ratio)
            
            # 4. 应用先验信息修改更新
            update = update * prior
            
            # 5. 更新估计
            estimate = estimate * update
            
            # 6. 确保非负
            estimate = np.maximum(estimate, 1e-10)
            
            # 7. 每10次迭代应用轻微平滑（仅在非峰区域）
            if iter % 10 == 0 and iter > 0:
                # 创建峰掩码
                peak_mask = np.zeros(length, dtype=bool)
                for peak_pos, peak_width, _ in potential_peaks:
                    start = max(0, peak_pos - peak_width)
                    end = min(length, peak_pos + peak_width + 1)
                    peak_mask[start:end] = True
                
                # 仅在非峰区域应用平滑
                smoothed = gaussian_filter1d(estimate, sigma=0.5)
                estimate[~peak_mask] = smoothed[~peak_mask]
            
            if iter % 10 == 0:
                print(f"  贝叶斯反卷积迭代: {iter}/{iterations}")
        
        # 峰值增强
        if self.peak_enhancement:
            print("  应用峰值增强...")
            for peak_pos, _, _ in potential_peaks:
                if 0 <= peak_pos < length:
                    # 增强峰值
                    estimate[peak_pos] *= 1.2
        
        return estimate
    
    def process_spectra(self, original_spectra, noisy_spectra):
        """处理多个能谱并返回反卷积结果"""
        deconvolved_spectra = []
        
        print("开始反卷积处理...")
        print(f"使用算法: {self.algorithm}")
        start_time = time.time()
        
        for i in range(len(noisy_spectra)):
            print(f"处理能谱 {i+1}/{len(noisy_spectra)}")
            
            # 根据选择的算法执行反卷积
            if self.algorithm == 'hybrid':
                deconvolved = self.hybrid_deconvolution(noisy_spectra[i])
            elif self.algorithm == 'sparse':
                deconvolved = self.sparse_deconvolution(noisy_spectra[i])
            elif self.algorithm == 'blind':
                deconvolved = self.blind_deconvolution(noisy_spectra[i])
            elif self.algorithm == 'admm':
                deconvolved = self.admm_deconvolution(noisy_spectra[i])
            elif self.algorithm == 'bayesian':
                deconvolved = self.bayesian_deconvolution(noisy_spectra[i])
            elif self.algorithm == 'adaptive':
                deconvolved = self.adaptive_deconvolution(noisy_spectra[i])
            else:
                print(f"未知算法: {self.algorithm}，使用默认的hybrid算法")
                deconvolved = self.hybrid_deconvolution(noisy_spectra[i])
            
            deconvolved_spectra.append(deconvolved)
        
        print(f"反卷积处理完成，耗时 {time.time() - start_time:.2f} 秒")
        
        return deconvolved_spectra
    
    def visualize_results(self, x, original_spectra, noisy_spectra, deconvolved_spectra, output_file="deconvolution_results.html"):
        """可视化反卷积结果"""
        fig = make_subplots(rows=len(original_spectra), cols=1, 
                           subplot_titles=[f"能谱 {i+1} 反卷积结果" for i in range(len(original_spectra))],
                           vertical_spacing=0.05)
        
        for i in range(len(original_spectra)):
            fig.add_trace(
                go.Scatter(x=x, y=original_spectra[i], mode='lines', name='原始能谱', 
                          line=dict(color='blue')),
                row=i+1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=noisy_spectra[i], mode='lines', name='卷积后带噪声能谱', 
                          line=dict(color='red')),
                row=i+1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=deconvolved_spectra[i], mode='lines', name='反卷积结果', 
                          line=dict(color='green')),
                row=i+1, col=1
            )
        
        fig.update_layout(height=500*len(original_spectra), width=1000, 
                         title="能谱反卷积结果",
                         showlegend=False)
        
        fig.write_html(output_file)
        print(f"反卷积结果已保存至 {output_file}")
        
        return fig


def main(num_spectra=10, rev=0.1, error_amplitude=1.0, use_wiener=False, snr=100.0, 
         peak_enhancement=True, algorithm='hybrid', peak_intensity=30000):
    print(f"参数设置: 能谱数量={num_spectra}, 分辨率参数={rev}, 误差强度={error_amplitude}, 特征峰强度={peak_intensity}")
    if use_wiener:
        print(f"启用Wiener滤波, 信噪比={snr}")
    if peak_enhancement:
        print("启用峰增强模式")
    print(f"使用反卷积算法: {algorithm}")
    
    # 生成能谱数据
    generator = EnergySpectrumGenerator(num_spectra=num_spectra, rev=rev, 
                                       error_amplitude=error_amplitude,
                                       peak_intensity=peak_intensity)
    original_spectra, convolved_spectra, noisy_spectra = generator.generate_data()
    generator.visualize_html("energy_spectra.html")
    
    # 反卷积处理
    deconvolver = DeconvolutionProcessor(rev=rev, use_wiener=use_wiener, snr=snr, 
                                        peak_enhancement=peak_enhancement, algorithm=algorithm)
    deconvolved_spectra = deconvolver.process_spectra(original_spectra, noisy_spectra)
    deconvolver.visualize_results(generator.x, original_spectra, noisy_spectra, deconvolved_spectra, "deconvolution_results.html")
    
    print("所有处理完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="能谱生成与反卷积处理")
    parser.add_argument("--num_spectra", type=int, default=10, help="生成的能谱数量")
    parser.add_argument("--rev", type=float, default=0.01, help="分辨率参数，影响高斯卷积的宽度")
    parser.add_argument("--error_amplitude", type=float, default=1.0, help="统计误差强度参数")
    parser.add_argument("--use_wiener", action="store_true", help="是否使用Wiener滤波")
    parser.add_argument("--snr", type=float, default=100.0, help="Wiener滤波的信噪比参数")
    parser.add_argument("--no_peak_enhancement", action="store_true", help="禁用峰增强模式")
    parser.add_argument("--algorithm", type=str, default="hybrid", 
                       choices=["hybrid", "sparse", "blind", "admm", "bayesian", "adaptive"],
                       help="选择反卷积算法")
    parser.add_argument("--peak_intensity", type=float, default=30000.0, help="特征峰强度参数")
    
    args = parser.parse_args()
    
    main(num_spectra=args.num_spectra, rev=args.rev, error_amplitude=args.error_amplitude,
         use_wiener=args.use_wiener, snr=args.snr, peak_enhancement=not args.no_peak_enhancement,
         algorithm=args.algorithm, peak_intensity=args.peak_intensity) 