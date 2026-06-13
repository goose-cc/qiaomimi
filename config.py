# config.py
import os

class Config:
    def __init__(self):
        # =========================
        # 新的数据生成配置（用于 toy model clean data）
        # =========================
        self.random_seed = 2026

        # 每个模型构造 100000 对纯净数据
        self.num_clean_samples_per_model = 100000

        # x / y 采样点数
        self.num_x_points = 100
        self.num_y_points = 100

        # x, y 区间
        self.x_min = 0.0
        self.x_max = 2.0
        self.y_min = 3.0
        self.y_max = 8.0

        # 叠加图中实际显示多少条浅色曲线
        self.overlay_plot_samples = 500

        # 输出目录
        self.output_root = './toy_clean_data'
        self.figure_root = './toy_clean_figures'

        # Model 3 固定参数
        self.model3_m = 0.8
        self.model3_gamma = 0.5

        # a1, a2 波动水平（沿用你项目里 generate_dataset_100k.py 的思路）
        self.error_levels = [0.30, 0.10, 0.01]

        # =========================
        # 原训练配置（保留）
        # =========================
       # =========================
# 训练配置：通过环境变量 EXP 选择实验
# exp1：旧 data，三模型纯净混合数据
# exp2：Model1 noisy 单独训练
# exp3：先加载 exp1 模型，再用 Model1 noisy 微调
# =========================
        self.exp = os.environ.get("EXP", "exp1")

        if self.exp == "exp1":
            self.train_dir = "./data/train.npz"
            self.val_dir = "./data/val.npz"
            self.test_dir = "./data/test.npz"
            self.model_path = "./model/exp1_clean_three_models.pth"

        elif self.exp == "exp2":
            self.train_dir = "./data_exp2/train.npz"
            self.val_dir = "./data_exp2/val.npz"
            self.test_dir = "./data_exp2/test.npz"
            self.model_path = "./model/exp2_model1_noisy.pth"

        elif self.exp == "exp3":
            self.train_dir = "./data_exp2/train.npz"
            self.val_dir = "./data_exp2/val.npz"
            self.test_dir = "./data_exp2/test.npz"
            self.model_path = "./model/exp3_finetune_model1_noisy.pth"

        else:
            raise ValueError(f"未知实验类型: {self.exp}")

        self.noise_level = 0.01

        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 3
        self.batch_size = 32
        self.loss_profile = "base"


