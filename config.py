# config.py 文件内容示例
class Config:
    def __init__(self):
        # 数据参数
        self.num_train_samples = 1000  # 训练数据中有多少个波（每个波是一条测试数据）
        self.train_dir = './train'
        self.num_val_samples = 100      # 验证数据中有多少个波
        self.val_dir = './val'
        self.num_test_samples = 100     # 测试数据中有多少个波
        self.test_dir = './test'
        
        # 每一个fx中x的采样点数, gy中y的采样点数
        self.num_x_points = 100
        self.num_y_points = 100
        self.noise_level = 0.01

        # 训练配置
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs =50
        self.batch_size = 32

        self.model_path = './model/best_model.pth'  # 模型保存路径