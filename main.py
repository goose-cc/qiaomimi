from torch.utils.data import DataLoader
from config import Config
from PIDataset import PeakInversionDataset
from ModelTools import ModelTools
from utils.visualize import visualize_prediction

import argparse
import os
from datetime import datetime


LOSS_PROFILES = [
    "base",
    "mse_grad",
    "huber_grad",
    "pinn",
    "pinn_smooth",
    "pinn_tikhonov",
    "pinn_huber",
    "pinn_full",
]


def parse_args():
    parser = argparse.ArgumentParser(description="峰值反演模型训练与评估")

    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["cnn", "unet", "pinn", "transformer"],
        help="选择网络结构: cnn, unet, pinn 或 transformer",
    )

    parser.add_argument(
        "--loss_profile",
        type=str,
        default=None,
        choices=LOSS_PROFILES,
        help="选择损失组合: base, mse_grad, pinn, pinn_smooth 等；不填则使用 config.loss_profile",
    )

    # 兼容旧参数；现在真正生效的是 loss_profile
    parser.add_argument(
        "--train_type",
        type=str,
        default=None,
        choices=["supervised", "pinn"],
        help="兼容旧参数。supervised 默认对应 base，pinn 默认对应 config.loss_profile",
    )

    parser.add_argument(
        "--load_model",
        action="store_true",
        help="是否加载已有模型并直接测试",
    )

    # 可选参数：保留兼容性。
    # 正常实验三不需要手动传，config.py 里 self.exp='exp3' 会自动加载预训练模型。
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="可选：手动指定预训练模型路径；不填则使用 config.pretrain_model_path",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="用于预测的样本索引",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="覆盖 config.max_train_samples",
    )

    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="覆盖 config.max_val_samples",
    )

    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="覆盖 config.max_test_samples",
    )

    return parser.parse_args()


def create_training_result_dir(model_type, loss_profile):
    """
    在 picture/training_results 下自动创建本次训练结果文件夹。

    命名格式：
    train_001_20260611_1430_cnn_base
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))

    picture_root = os.path.join(project_dir, "picture", "training_results")
    os.makedirs(picture_root, exist_ok=True)

    existing_runs = [
        name
        for name in os.listdir(picture_root)
        if os.path.isdir(os.path.join(picture_root, name))
        and name.startswith("train_")
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
    """
    run_info_path = os.path.join(config.result_dir, "run_info.txt")

    with open(run_info_path, "w", encoding="utf-8") as f:
        f.write(f"训练编号: 第 {config.run_id} 次\n")
        f.write(f"训练时间: {config.run_time}\n")
        f.write(f"实验类型: {getattr(config, 'exp', 'unknown')}\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"loss_profile: {getattr(config, 'loss_profile', 'default')}\n")
        f.write(f"num_epochs: {config.num_epochs}\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"learning_rate: {config.learning_rate}\n")
        f.write(f"weight_decay: {config.weight_decay}\n")
        f.write(f"model_path: {config.model_path}\n")
        f.write(f"pretrain_model_path: {getattr(config, 'pretrain_model_path', None)}\n")
        f.write(f"train_path: {getattr(config, 'train_path', getattr(config, 'train_dir', ''))}\n")
        f.write(f"val_path: {getattr(config, 'val_path', getattr(config, 'val_dir', ''))}\n")
        f.write(f"test_path: {getattr(config, 'test_path', getattr(config, 'test_dir', ''))}\n")
        f.write(f"max_train_samples: {getattr(config, 'max_train_samples', None)}\n")
        f.write(f"max_val_samples: {getattr(config, 'max_val_samples', None)}\n")
        f.write(f"max_test_samples: {getattr(config, 'max_test_samples', None)}\n")
        f.write(f"normalize_gy_per_sample: {getattr(config, 'normalize_gy_per_sample', False)}\n")

    print(f"训练信息已保存到: {run_info_path}")


def main():
    config = Config()
    args = parse_args()

    model_type = args.model_type

    # =====================
    # loss profile 设置
    # =====================
    if args.loss_profile is not None:
        loss_profile = args.loss_profile
    elif args.train_type == "supervised":
        loss_profile = "base"
    else:
        loss_profile = config.loss_profile

    # 同步回 config，保证 ModelTools / losses.py 能读到一致配置
    config.loss_profile = loss_profile

    # =====================
    # 本次训练结果目录
    # =====================
    result_dir, run_id, time_str = create_training_result_dir(model_type, loss_profile)
    config.result_dir = result_dir
    config.run_id = run_id
    config.run_time = time_str

    print(f"本次训练编号: 第 {config.run_id} 次")
    print(f"本次训练时间: {config.run_time}")
    print(f"训练结果保存目录: {config.result_dir}")

    save_run_info(config, args)

    # =====================
    # 样本数设置
    # =====================
    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else config.max_train_samples
    )
    max_val_samples = (
        args.max_val_samples
        if args.max_val_samples is not None
        else config.max_val_samples
    )
    max_test_samples = (
        args.max_test_samples
        if args.max_test_samples is not None
        else config.max_test_samples
    )

    # =====================
    # 数据集
    # =====================
    train_dataset = PeakInversionDataset(
        config.train_dir,
        max_samples=max_train_samples,
        normalize_gy=config.normalize_gy_per_sample,
    )

    val_dataset = PeakInversionDataset(
        config.val_dir,
        max_samples=max_val_samples,
        normalize_gy=config.normalize_gy_per_sample,
    )

    test_dataset = PeakInversionDataset(
        config.test_dir,
        max_samples=max_test_samples,
        normalize_gy=config.normalize_gy_per_sample,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    predict_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # =====================
    # 模型工具
    # =====================
    modeltools = ModelTools(
        config,
        model_type=model_type,
        loss_profile=loss_profile,
        LoadModel=args.load_model,
    )

    # =====================
    # 实验三：自动加载预训练模型
    # 优先使用命令行 --pretrain_model；
    # 如果没有传，则使用 config.pretrain_model_path。
    # =====================
    if args.pretrain_model is not None:
        pretrain_model_path = args.pretrain_model
    else:
        pretrain_model_path = getattr(config, "pretrain_model_path", None)

    if pretrain_model_path is not None and not args.load_model:
        if not os.path.exists(pretrain_model_path):
            raise FileNotFoundError(f"找不到预训练模型: {pretrain_model_path}")

        print(f"加载预训练模型作为初始权重: {pretrain_model_path}")
        modeltools.load_model(pretrain_model_path)
        print("预训练模型加载完成，开始微调训练。")

    # =====================
    # 训练
    # =====================
    if not args.load_model:
        if pretrain_model_path is None:
            print("不加载预训练模型，开始训练...")
        else:
            print("开始微调训练...")

        modeltools.train(train_loader, val_loader, num_epochs=config.num_epochs)

        # 训练后尝试加载最优模型
        try:
            modeltools.load_model(modeltools.model_path)
        except Exception:
            pass

    # =====================
    # 单样本预测可视化
    # =====================
    gy, fx_true = test_dataset.__getitem__(args.index)
    x = test_dataset.__getx__(args.index)

    fx_pred, loss = modeltools.predict(
        predict_loader=predict_loader,
        index=args.index,
    )

    print(f"Predicted fx shape: {fx_pred.shape}")
    print(f"Ground truth fx shape: {fx_true.shape}")
    print(f"Prediction Loss MSE: {loss:.6f}")
    print(f"Predicted fx: {fx_pred}")

    visualize_prediction(
        fx_true.squeeze().numpy(),
        fx_pred.squeeze().detach().cpu().numpy(),
        x,
    )

    test_loss_mse = modeltools.validate(test_loader)
    print(f"Test Loss MSE: {test_loss_mse:.6f}")


if __name__ == "__main__":
    main()
