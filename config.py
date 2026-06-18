# config.py

class Config:
    def __init__(self):
        # =====================
        # 实验选择
        # exp1：三模型纯净数据训练
        # exp2：Model1 noisy 单独训练
        # exp3：加载 exp1 模型后，在 Model1 noisy 上微调
        #
        # 队友运行命令仍然是：
        # python main.py --model_type cnn --index 20
        # =====================
        self.exp = "exp1"  # 想跑实验2就改成 "exp2"，想跑实验3就改成 "exp3"

        if self.exp == "exp1":
            self.train_path = "./data/train.npz"
            self.val_path = "./data/val.npz"
            self.test_path = "./data/test.npz"
            self.model_path = "./model/exp1_clean_three_models.pth"
            self.pretrain_model_path = None

        elif self.exp == "exp2":
            self.train_path = "./data_exp2/train.npz"
            self.val_path = "./data_exp2/val.npz"
            self.test_path = "./data_exp2/test.npz"
            self.model_path = "./model/exp2_model1_noisy.pth"
            self.pretrain_model_path = None

        elif self.exp == "exp3":
            self.train_path = "./data_exp2/train.npz"
            self.val_path = "./data_exp2/val.npz"
            self.test_path = "./data_exp2/test.npz"
            self.model_path = "./model/exp3_finetune_model1_noisy.pth"
            self.pretrain_model_path = "./model/exp1_clean_three_models.pth"

        else:
            raise ValueError(f"未知实验类型: {self.exp}")

        # 兼容 main.py / 原代码中使用 train_dir / val_dir / test_dir 的写法
        self.train_dir = self.train_path
        self.val_dir = self.val_path
        self.test_dir = self.test_path

        # =====================
        # toy model 数据生成配置
        # 用于 data_generate.py / data_generate_model1_noisy.py
        # =====================
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

        # 数据生成输出目录
        self.output_root = "./toy_clean_data"
        self.figure_root = "./toy_clean_figures"

        # Model 3 固定参数
        self.model3_m = 0.8
        self.model3_gamma = 0.5

        # a1, a2 波动水平
        self.error_levels = [0.30, 0.10, 0.01]

        # =====================
        # 数据读取配置
        # =====================
        # 是否对输入 g(y) 做逐样本标准化
        # 建议 False，因为 g(y) 的幅值信息对反演 f(x) 有意义
        self.normalize_gy_per_sample = False

        # =====================
        # 调试用样本数
        # CPU 或调试时建议小样本；正式实验可改成 None
        # =====================
        self.max_train_samples = 5000
        self.max_val_samples =800
        self.max_test_samples = 800   

        # 正式完整训练时可以改成：
        # self.max_train_samples = None
        # self.max_val_samples = None
        # self.max_test_samples = None

        # =====================
        # 训练配置
        # =====================
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 20
        self.batch_size = 64

        # =====================
        # Loss 实验配置
        # 可选：
        # base, mse_grad, huber_grad, pinn, pinn_smooth,
        # pinn_tikhonov, pinn_huber, pinn_full
        # =====================
        self.loss_profile = "base"

        self.lambda_grad = 0.1
        self.lambda_physics = 0.01
        self.lambda_smooth = 0.01
        self.lambda_tv = 0.001
        self.lambda_tikhonov = 0.0001

        # =====================
        # Transformer 参数
        # =====================
        self.transformer_d_model = 64
        self.transformer_nhead = 4
        self.transformer_num_layers = 3
        self.transformer_dim_feedforward = 128
        self.transformer_dropout = 0.1

        # =====================
        # 其他配置
        # =====================
        self.model_dir = "./model"
        self.noise_level = 0.01
