from torch.utils.data import DataLoader
from config import Config
from PIDataset import PeakInversionDataset
from ModelTools import ModelTools
from utils.visualize import visualize_prediction
import argparse
import torch

def parse_args():

    parser = argparse.ArgumentParser(description="峰值反演模型训练与评估")
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'unet'],
                        help='选择模型类型: cnn 或 unet')
    parser.add_argument('--load_model', action='store_true',
                        help='是否加载预训练模型')
    parser.add_argument('--index', type=int, default=0,
                        help='用于预测的样本索引')
    return parser.parse_args()

def main():
    # 配置
    config = Config()

    args = parse_args()
    model_type = args.model_type
    load_model = args.load_model
    index = args.index
    
    # 创建数据集
    train_dataset = PeakInversionDataset(config.train_path)
    val_dataset = PeakInversionDataset(config.val_path)
    test_dataset = PeakInversionDataset(config.test_path)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    predict_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # 训练模型
    modeltools = ModelTools(config, model_type=model_type, LoadModel=load_model)  # 或 'unet'
    
    # 如果需要训练，取消注释下面的行
    if not load_model:
        print("不加载预训练模型，开始训练...")
        modeltools.train(train_loader, val_loader, num_epochs=config.num_epochs)
    
    # 可视化一些结果
    gcy_noisy, fx_true = test_dataset.__getitem__(index)
    x = test_dataset.__getx__(index)
    fx_pred, loss = modeltools.predict(predict_loader=predict_loader, index=index)
    print(f'Predicted fx shape: {fx_pred.shape}')
    print(f'Ground truth fx shape: {fx_true.shape}')
    print(f'Prediction Loss MSE: {loss:.6f}')
    print(f'Predicted fx: {fx_pred}')
    visualize_prediction(fx_true.squeeze().numpy(), fx_pred.squeeze().detach().cpu().numpy(), x)

    # 测试集评估
    test_loss_mse = modeltools.validate(test_loader)
    print(f'Test Loss MSE: {test_loss_mse:.6f}')

if __name__ == "__main__":
    main()
