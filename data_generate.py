import numpy as np
import os
from scipy.special import voigt_profile
from config import Config

# 创建配置信息对象
cfg = Config()

def gaussian(x, A, mu, sigma):
    """高斯峰

    Args:
        x: 自变量区间（实质上是用离散点来表征区间）
        A: 波峰振幅
        mu: 均值
        sigma: 方差

    Return:
        x 中每个离散点对应的 y 值
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def lorentzian(x, A, mu, gamma):
    """洛伦兹峰 (BW峰)

    Args:
        x: 自变量区间（实质上是用离散点来表征区间）
        A: 波峰振幅
        mu: 均值
        gamma: 方差
    
    Return:
        x 中每个离散点对应的 y 值
    """
    return A * gamma**2 / ((x - mu)**2 + gamma**2) / (np.pi * gamma)

def voigt(x, A, mu, sigma, gamma):
    """Voigt峰 （高斯-洛伦兹卷积)

    Args:
        x: 自变量区间（实质上是用离散点来表征区间）
        A: 波峰振幅
        mu: 均值
        sigma: 高斯峰方差
        gamma: 洛伦兹峰方差
    
    Return:
        x 中每个离散点对应的 y 值
    """
    # 调用 python 库函数中的 vogit_profile，这个函数默认均值为 0，
    # 因此需要通过 x - mu 把输入 x 变换到均值为 0
    return A * voigt_profile(x - mu, sigma, gamma)

def generate_fx(x, num_peaks=None):
    """生成包含多种峰结构的f(x)

    Args:
        x: 自变量区间（实质上是用离散点来表征区间）
        num_peaks: 生成的峰的数量

    Return:
        x 中每个离散点对应的 y 值
    """
    if num_peaks is None:
        # 随机生成峰的个数
        num_peaks = np.random.randint(1, 5)
    
    # 生成与 x 结构完全相同的全 0 量
    fx = np.zeros_like(x)
    peak_types = ['gaussian', 'lorentzian', 'voigt']
    
    for _ in range(num_peaks):
        peak_type = np.random.choice(peak_types, p=[0.4, 0.4, 0.2])
        
        if peak_type == 'gaussian':
            # uniform 是利用均匀分布在指定区间内生成一个随机数的函数
            A = np.random.uniform(0.5, 5.0)
            mu = np.random.uniform(0.1, 1.9)
            sigma = np.random.uniform(0.05, 0.3)
            peak = gaussian(x, A, mu, sigma)
        
        elif peak_type == 'lorentzian':
            A = np.random.uniform(0.5, 5.0)
            mu = np.random.uniform(0.1, 1.9)
            gamma = np.random.uniform(0.05, 0.5)
            peak = lorentzian(x, A, mu, gamma)
        
        else:  # voigt
            A = np.random.uniform(0.5, 5.0)
            mu = np.random.uniform(0.1, 1.9)
            sigma = np.random.uniform(0.02, 0.2)
            gamma = np.random.uniform(0.02, 0.3)
            peak = voigt(x, A, mu, sigma, gamma)
        
        fx += peak
    
    # 添加基底噪声
    #fx += np.random.normal(0, 0.01, size=len(x))
    return np.clip(fx, 0, None)  # 确保非负

def compute_gy(x, fx, y_points):
    """计算积分方程 g(y) = ∫f(x)/(y-x)dx (使用向量化方法)
    计算 y_points 上的 g(y)
    
    Args:
        x: x的区间, 定义了积分区间的采样范围

        fx: 波峰函数

        y_points: y的区间

    Return:
        y_points 对应的 g(y_points) 向量
    
    """
    # 创建网格以进行向量化计算
    X, Y = np.meshgrid(x, y_points)
    
    # 计算分母并避免奇点
    denominator = Y - X
    # 如果 |y-x| < 1e-6（非常接近），设为 1e-6，否则保持原值
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)
    
    # 计算被积函数
    integrand = fx / denominator
    
    # 使用梯形法进行数值积分（手动实现）
    dx = x[1] - x[0] if len(x) > 1 else 1
    gy = np.trapezoid(integrand, x, axis=1)
    
    return gy

def add_noise(gy, noise_level=0.01):
    """添加相对噪声"""
    # 使用与信号幅度成比例的高斯噪声
    noise = np.random.normal(0, noise_level * np.abs(gy))
    return gy + noise

def generate_data(num_samples, data_dir):
    '''
    生成数据
    
    :param num_samples: 生成的数据条目数量
    :param data_dir: 存储数据的目录
    :return
    '''
    os.makedirs(data_dir, exist_ok=True)    # 创建目录
    
    # 定义离散点, x, y本质上是fx和gy这两个不同函数的定义区间，所以他们之间的采样密度是无关的
    # num_x_points: x 在其定义区间上的采样数量
    # num_y_points: y 在其定义区间上的采样数量
    x = np.linspace(0, 2, cfg.num_x_points)
    y = np.linspace(2.1, 10, cfg.num_y_points)
    
    for i in range(num_samples):
        # 生成f(x)
        fx = generate_fx(x)
        
        # 计算g(y)
        gy_clean = compute_gy(x, fx, y)
        
        # 添加噪声
        gy_noisy = add_noise(gy_clean, cfg.noise_level)
        
        # 保存
        np.savez(
            os.path.join(data_dir, f"sample_{i:05d}.npz"),
            x=x, y=y, fx=fx, gy_clean=gy_clean, gy_noisy=gy_noisy
        )
        
        if (i+1) % 1000 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    print(f"Data generation for {data_dir} completed")

if __name__ == "__main__":
    generate_data(cfg.num_train_samples, cfg.train_dir)
    generate_data(cfg.num_val_samples, cfg.val_dir)
    generate_data(cfg.num_test_samples, cfg.test_dir)