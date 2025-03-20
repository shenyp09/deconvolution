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
    def __init__(self, length=8192, num_spectra=10, rev=0.01, error_amplitude=1.0):
        self.length = length
        self.num_spectra = num_spectra
        self.rev = rev  # 高斯卷积的分辨率参数
        self.error_amplitude = error_amplitude  # 统计误差强度参数
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
        
        fig.update_layout(height=300*self.num_spectra, width=1000, 
                         title="能谱数据集可视化",
                         showlegend=False)
        
        fig.write_html(output_file)
        print(f"可视化结果已保存至 {output_file}")
        
        return fig


class DeconvolutionProcessor:
    def __init__(self, rev=0.01):
        self.rev = rev  # 卷积参数
        
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
    
    def process_spectra(self, original_spectra, noisy_spectra):
        """处理多个能谱并返回反卷积结果"""
        deconvolved_spectra = []
        
        print("开始反卷积处理...")
        start_time = time.time()
        
        for i in range(len(noisy_spectra)):
            print(f"处理能谱 {i+1}/{len(noisy_spectra)}")
            # 使用优化的反卷积算法
            deconvolved = self.optimize_deconvolution(noisy_spectra[i])
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
        
        fig.update_layout(height=300*len(original_spectra), width=1000, 
                         title="能谱反卷积结果",
                         showlegend=False)
        
        fig.write_html(output_file)
        print(f"反卷积结果已保存至 {output_file}")
        
        return fig


def main(num_spectra=10, rev=0.01, error_amplitude=1.0):
    print(f"参数设置: 能谱数量={num_spectra}, 分辨率参数={rev}, 误差强度={error_amplitude}")
    
    # 生成能谱数据
    generator = EnergySpectrumGenerator(num_spectra=num_spectra, rev=rev, error_amplitude=error_amplitude)
    original_spectra, convolved_spectra, noisy_spectra = generator.generate_data()
    generator.visualize_html("energy_spectra.html")
    
    # 反卷积处理
    deconvolver = DeconvolutionProcessor(rev=rev)
    deconvolved_spectra = deconvolver.process_spectra(original_spectra, noisy_spectra)
    deconvolver.visualize_results(generator.x, original_spectra, noisy_spectra, deconvolved_spectra, "deconvolution_results.html")
    
    print("所有处理完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="能谱生成与反卷积处理")
    parser.add_argument("--num_spectra", type=int, default=10, help="生成的能谱数量")
    parser.add_argument("--rev", type=float, default=0.01, help="分辨率参数，影响高斯卷积的宽度")
    parser.add_argument("--error_amplitude", type=float, default=1.0, help="统计误差强度参数")
    
    args = parser.parse_args()
    
    main(num_spectra=args.num_spectra, rev=args.rev, error_amplitude=args.error_amplitude) 