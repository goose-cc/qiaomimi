# config.py
class Config:
    def __init__(self):
        # =====================
        # 数据路径：队友新生成的集中式 npz 数据集
        # =====================
        self.train_path = './data/train.npz'
        self.val_path = './data/val.npz'
        self.test_path = './data/test.npz'

        # 为兼容原代码，保留 train_dir / val_dir / test_dir 名称
        self.train_dir = self.train_path
        self.val_dir = self.val_path
        self.test_dir = self.test_path

        # =====================
        # 网格参数
        # generate_dataset_100k.py 中使用：x in [0, 2], y in [3, 8]
        # =====================
        self.num_x_points = 100
        self.num_y_points = 100
        self.x_min = 0.0
        self.x_max = 2.0
        self.y_min = 3.0
        self.y_max = 8.0

        # 是否对输入 g(y) 做逐样本标准化
        # 注意：如果改成 False，losses.py 中 physics loss 也会自动不做标准化比较
        self.normalize_gy_per_sample = False

        # =====================
        # 调试用样本数
        # CPU 或调试时建议用小样本；正式实验可改为 None 使用完整数据
        # =====================
        self.max_train_samples = 2000
        self.max_val_samples = 500
        self.max_test_samples = 500

        # 如果你已经装好了 CUDA 版 PyTorch，可以改成：
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
        # 可选：base, mse_grad, huber_grad, pinn, pinn_smooth, pinn_tikhonov, pinn_huber, pinn_full
        # =====================
        self.loss_profile = 'base'

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

        # 模型保存目录
        self.model_dir = './model'
        self.model_path = './model/best_model.pth'
