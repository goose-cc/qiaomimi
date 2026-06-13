import os
from datetime import datetime

from torch.utils.data import DataLoader
from config import Config
from PIDataset import PeakInversionDataset
from ModelTools import ModelTools
from utils.visualize import visualize_prediction
import argparse
import torch

def parse_args():

    parser = argparse.ArgumentParser(description="峰值反演模型训练与评估")
    parser.add_argument('--model_type', type=str, default='cnn',
                    choices=['cnn', 'unet', 'pinn', 'transformer'],
                    help='模型类型: cnn, unet, pinn 或 transformer')
    parser.add_argument('--load_model', action='store_true',
                        help='是否加载预训练模型')
    parser.add_argument('--pretrain_model', type=str, default=None,
                    help='加载已有模型作为初始权重，然后继续训练，用于实验三微调')

    parser.add_argument('--index', type=int, default=0,
                        help='用于预测的样本索引')
    parser.add_argument(
    '--loss_profile',
    type=str,
    default=None,
    choices=['base', 'mse_grad', 'pinn', 'pinn_smooth', 'pinn_tikhonov', 'pinn_full'],
    help='loss 组合方式'
)

    return parser.parse_args()

def create_training_result_dir(model_type, loss_profile):
    """
    在 big_create/picture 下自动创建本次训练结果文件夹。

    命名格式：
    train_001_20260606_1430_transformer_base
    """
    # 当前文件 main.py 所在目录：calc-k-master
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # 上一级目录：big_create
    big_create_dir = os.path.dirname(project_dir)

    # 统一保存训练结果的目录：big_create/picture
    picture_root = os.path.join(project_dir, "picture", "training_results")
    os.makedirs(picture_root, exist_ok=True)

    existing_runs = [
        name for name in os.listdir(picture_root)
        if os.path.isdir(os.path.join(picture_root, name)) and name.startswith("train_")
    ]

    run_id = len(existing_runs) + 1
    time_str = datetime.now().strftime("%Y%m%d_%H%M")

    if loss_profile is None:
        loss_profile = "default"

    run_name = f"train_{run_id:03d}_{time_str}_{model_type}_{loss_profile}"
    result_dir = os.path.join(picture_root, run_name)

    os.makedirs(result_dir, exist_ok=True)

    return result_dir, run_id, time_str


def save_run_info(config, args):
    """
    保存本次训练的基本信息。
    这个函数会自动创建 run_info.txt，不需要手动新建。
    """
    run_info_path = os.path.join(config.result_dir, "run_info.txt")

    with open(run_info_path, "w", encoding="utf-8") as f:
        f.write(f"训练编号: 第 {config.run_id} 次\n")
        f.write(f"训练时间: {config.run_time}\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"loss_profile: {getattr(config, 'loss_profile', 'default')}\n")
        f.write(f"num_epochs: {config.num_epochs}\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"learning_rate: {config.learning_rate}\n")
        f.write(f"weight_decay: {config.weight_decay}\n")
        f.write(f"model_path: {config.model_path}\n")
        f.write(f"pretrain_model: {args.pretrain_model}\n")

        f.write(f"train_path: {getattr(config, 'train_path', getattr(config, 'train_dir', ''))}\n")
        f.write(f"val_path: {getattr(config, 'val_path', getattr(config, 'val_dir', ''))}\n")
        f.write(f"test_path: {getattr(config, 'test_path', getattr(config, 'test_dir', ''))}\n")

    print(f"训练信息已保存到: {run_info_path}")


def main():
    # 配置
    config = Config()
    args = parse_args()

    if args.loss_profile is not None:
        config.loss_profile = args.loss_profile

    result_dir, run_id, time_str = create_training_result_dir(
        model_type=args.model_type,
        loss_profile=getattr(config, "loss_profile", "default")
    )

    config.result_dir = result_dir
    config.run_id = run_id
    config.run_time = time_str

    print(f"本次训练编号: 第 {run_id} 次")
    print(f"本次训练时间: {time_str}")
    print(f"训练结果保存目录: {result_dir}")

    save_run_info(config, args)

    model_type = args.model_type
    load_model = args.load_model
    index = args.index
    
    # 创建数据集
    train_dataset = PeakInversionDataset(config.train_dir)
    val_dataset = PeakInversionDataset(config.val_dir)
    test_dataset = PeakInversionDataset(config.test_dir)
    
    # 创建数据加载器 - Windows 上必须使用 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size
, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size
, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size
, shuffle=False, num_workers=0, pin_memory=False)
    predict_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # 训练模型
    modeltools = ModelTools(config, model_type=model_type, LoadModel=load_model)  # 或 'unet'
    
    # 实验三：加载实验一训练好的模型，然后继续训练
    if args.pretrain_model is not None:
        if load_model:
            raise ValueError("不要同时使用 --load_model 和 --pretrain_model。实验三微调只需要 --pretrain_model。")

        if not os.path.exists(args.pretrain_model):
            raise FileNotFoundError(f"找不到预训练模型: {args.pretrain_model}")

        print(f"加载预训练模型作为初始权重: {args.pretrain_model}")
        modeltools.load_model(args.pretrain_model)
        print("预训练模型加载完成，开始后续微调训练。")


    # 如果需要训练，取消注释下面的行
    if not load_model:
        print("不加载预训练模型，开始训练...")
        modeltools.train(train_loader, val_loader, num_epochs=config.num_epochs)
        # 训练完成后加载保存的最佳模型再做预测与评估，避免使用最后一次未必最优的权重
        try:
            modeltools.load_model(config.model_path)
        except Exception:
            pass
    
    # 可视化一些结果
    gcy_noisy, fx_true = test_dataset.__getitem__(index)
    x = test_dataset.__getx__(index)
    fx_pred, loss = modeltools.predict(predict_loader=predict_loader, index=index)
    print(f'Predicted fx shape: {fx_pred.shape}')
    print(f'Ground truth fx shape: {fx_true.shape}')
    print(f'Prediction Loss MSE: {loss.item():.6f}')

    print(f'Predicted fx: {fx_pred}')
    save_path = os.path.join(
        config.result_dir,
        f"prediction_index_{args.index}.png"
    )

    visualize_prediction(
        fx_true.squeeze().numpy(),
        fx_pred.squeeze().detach().cpu().numpy(),
        x,
        title=f"Run {config.run_id} | {args.model_type} | {getattr(config, 'loss_profile', 'default')} | index {args.index}",
        save_path=save_path
    )

    print(f"预测结果图已保存到: {save_path}")


    # 测试集评估
    test_loss_mse = modeltools.validate(test_loader)
    print(f'Test Loss MSE: {test_loss_mse:.6f}')

if __name__ == "__main__":
    main()
