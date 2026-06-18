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


EXP_CHOICES = ["exp1", "exp2", "exp3"]


def parse_args():
    parser = argparse.ArgumentParser(description="峰值反演模型训练与评估")

    # =====================
    # 实验 / 数据选择
    # =====================
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        choices=EXP_CHOICES,
        help=(
            "选择实验数据: "
            "exp1=三模型 clean 数据; "
            "exp2=Model1 noisy 单独训练; "
            "exp3=加载 exp1 模型后在 Model1 noisy 上微调。"
            "不传则使用 config.py 里的 config.exp。"
        ),
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["cnn", "unet", "transformer"],
        help="选择网络结构: cnn, unet 或 transformer。PINN 通过 --loss_profile 选择，不作为 model_type。",
    )

    parser.add_argument(
        "--loss_profile",
        type=str,
        default=None,
        choices=LOSS_PROFILES,
        help=(
            "选择损失组合: base, mse_grad, huber_grad, pinn, pinn_smooth, "
            "pinn_tikhonov, pinn_huber, pinn_full。"
            "不传则使用 config.loss_profile。"
        ),
    )

    # 兼容旧参数；现在真正生效的是 loss_profile
    parser.add_argument(
        "--train_type",
        type=str,
        default=None,
        choices=["supervised", "pinn"],
        help="兼容旧参数。supervised 默认对应 base；pinn 默认对应 config.loss_profile。",
    )

    # =====================
    # 训练控制
    # =====================
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="是否加载已有模型并直接测试，不训练。",
    )

    parser.add_argument(
        "--pretrain_model",
        type=str,
        default=None,
        help="可选：手动指定预训练模型路径；不填则 exp3 默认加载 exp1 的同网络 base 模型。",
    )

    parser.add_argument(
        "--pretrain_loss_profile",
        type=str,
        default="base",
        choices=LOSS_PROFILES,
        help="exp3 默认预训练模型使用的 loss_profile。默认 base。",
    )

    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="即使是 exp3，也不加载预训练模型。",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="覆盖 config.num_epochs。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="覆盖 config.batch_size。",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="覆盖 config.learning_rate。",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="覆盖 config.weight_decay。",
    )

    # =====================
    # 样本数量控制
    # =====================
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="使用全部 train/val/test 数据，等价于 max_train/val/test_samples=None。",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="覆盖 config.max_train_samples；调试时可设小，例如 2000。",
    )

    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="覆盖 config.max_val_samples；调试时可设小，例如 500。",
    )

    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="覆盖 config.max_test_samples；调试时可设小，例如 500。",
    )

    # =====================
    # 输出 / 预测
    # =====================
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="用于预测可视化的测试集样本索引。",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="可选：手动指定当前实验模型保存/加载路径。",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader 的 num_workers。Windows 建议保持 0。",
    )

    return parser.parse_args()


def apply_exp_config(config, exp):
    """
    根据命令行 --exp 切换数据路径。
    这样不需要手动修改 config.py 里的 self.exp。
    """
    config.exp = exp

    if exp == "exp1":
        config.train_path = "./data/train.npz"
        config.val_path = "./data/val.npz"
        config.test_path = "./data/test.npz"
        config.pretrain_model_path = None

    elif exp == "exp2":
        config.train_path = "./data_exp2/train.npz"
        config.val_path = "./data_exp2/val.npz"
        config.test_path = "./data_exp2/test.npz"
        config.pretrain_model_path = None

    elif exp == "exp3":
        config.train_path = "./data_exp2/train.npz"
        config.val_path = "./data_exp2/val.npz"
        config.test_path = "./data_exp2/test.npz"
        # 这里先设为 None，等 model_type/loss_profile 确定后再自动推导。
        config.pretrain_model_path = None

    else:
        raise ValueError(f"未知实验类型: {exp}")

    # 兼容 PIDataset / main.py 中使用 train_dir / val_dir / test_dir 的写法
    config.train_dir = config.train_path
    config.val_dir = config.val_path
    config.test_dir = config.test_path


def resolve_loss_profile(config, args):
    """
    决定本次使用哪个 loss_profile。
    优先级：
    1. 命令行 --loss_profile
    2. 旧参数 --train_type supervised -> base
    3. config.py 里的 config.loss_profile
    """
    if args.loss_profile is not None:
        return args.loss_profile

    if args.train_type == "supervised":
        return "base"

    return config.loss_profile


def build_checkpoint_path(config, model_type, loss_profile):
    """
    统一生成不会互相覆盖的 checkpoint 文件名。

    例如：
    ./model/exp1_transformer_base.pth
    ./model/exp1_transformer_pinn.pth
    ./model/exp2_transformer_base.pth
    ./model/exp2_transformer_pinn_huber.pth
    ./model/exp3_transformer_base.pth
    """
    model_dir = getattr(config, "model_dir", "./model")
    exp_name = getattr(config, "exp", "exp")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{exp_name}_{model_type}_{loss_profile}.pth")


def build_default_pretrain_path(config, model_type, pretrain_loss_profile="base"):
    """
    exp3 默认加载 exp1 的同网络模型作为预训练权重。
    默认使用 exp1 + base：
    ./model/exp1_transformer_base.pth
    """
    model_dir = getattr(config, "model_dir", "./model")
    return os.path.join(model_dir, f"exp1_{model_type}_{pretrain_loss_profile}.pth")


def apply_runtime_overrides(config, args):
    """
    应用命令行覆盖项。
    """
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay

    if args.full_data:
        config.max_train_samples = None
        config.max_val_samples = None
        config.max_test_samples = None


def resolve_sample_limits(config, args):
    """
    决定 train/val/test 实际使用多少样本。
    命令行 max_* 优先级高于 config.py。
    --full_data 已经在 apply_runtime_overrides 中把 config 的 max_* 设成 None。
    """
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

    return max_train_samples, max_val_samples, max_test_samples


def create_training_result_dir(exp, model_type, loss_profile):
    """
    在 picture/training_results 下自动创建本次训练结果文件夹。

    命名格式：
    train_001_20260611_1430_exp1_transformer_base
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

    run_name = f"train_{run_id:03d}_{time_str}_{exp}_{model_type}_{loss_profile}"
    result_dir = os.path.join(picture_root, run_name)

    os.makedirs(result_dir, exist_ok=True)

    return result_dir, run_id, time_str


def save_run_info(config, args, model_path, pretrain_model_path):
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
        f.write(f"model_path: {model_path}\n")
        f.write(f"pretrain_model_path: {pretrain_model_path}\n")
        f.write(f"train_path: {getattr(config, 'train_path', getattr(config, 'train_dir', ''))}\n")
        f.write(f"val_path: {getattr(config, 'val_path', getattr(config, 'val_dir', ''))}\n")
        f.write(f"test_path: {getattr(config, 'test_path', getattr(config, 'test_dir', ''))}\n")
        f.write(f"max_train_samples: {getattr(config, 'max_train_samples', None)}\n")
        f.write(f"max_val_samples: {getattr(config, 'max_val_samples', None)}\n")
        f.write(f"max_test_samples: {getattr(config, 'max_test_samples', None)}\n")
        f.write(f"normalize_gy_per_sample: {getattr(config, 'normalize_gy_per_sample', False)}\n")
        f.write(f"load_model: {args.load_model}\n")
        f.write(f"full_data: {args.full_data}\n")
        f.write(f"index: {args.index}\n")

    print(f"训练信息已保存到: {run_info_path}")


def check_required_paths(config, args, pretrain_model_path=None):
    """
    提前检查数据路径和预训练路径，避免训练开始后才报错。
    """
    required_data_paths = [
        ("train", config.train_dir),
        ("val", config.val_dir),
        ("test", config.test_dir),
    ]

    for name, path in required_data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 数据路径不存在: {path}")

    if pretrain_model_path is not None and not os.path.exists(pretrain_model_path):
        raise FileNotFoundError(f"找不到预训练模型: {pretrain_model_path}")


def main():
    config = Config()
    args = parse_args()

    # =====================
    # 1. 实验数据选择
    # =====================
    if args.exp is not None:
        apply_exp_config(config, args.exp)
    else:
        # 不传 --exp 时，沿用 config.py 里的 self.exp。
        # 同时确保 train_dir / val_dir / test_dir 与 train_path / val_path / test_path 一致。
        config.train_dir = config.train_path
        config.val_dir = config.val_path
        config.test_dir = config.test_path

    model_type = args.model_type

    # =====================
    # 2. loss profile 设置
    # =====================
    loss_profile = resolve_loss_profile(config, args)
    config.loss_profile = loss_profile

    # =====================
    # 3. 运行时覆盖训练参数
    # =====================
    apply_runtime_overrides(config, args)

    # =====================
    # 4. checkpoint 路径
    # =====================
    if args.model_path is not None:
        resolved_model_path = args.model_path
    else:
        resolved_model_path = build_checkpoint_path(config, model_type, loss_profile)

    config.model_path = resolved_model_path

    # =====================
    # 5. 预训练路径
    # =====================
    pretrain_model_path = None

    if not args.no_pretrain:
        if args.pretrain_model is not None:
            pretrain_model_path = args.pretrain_model
        elif getattr(config, "exp", None) == "exp3":
            pretrain_model_path = build_default_pretrain_path(
                config,
                model_type=model_type,
                pretrain_loss_profile=args.pretrain_loss_profile,
            )
        else:
            pretrain_model_path = getattr(config, "pretrain_model_path", None)

    config.pretrain_model_path = pretrain_model_path

    # =====================
    # 6. 本次训练结果目录
    # =====================
    result_dir, run_id, time_str = create_training_result_dir(
        exp=getattr(config, "exp", "unknown"),
        model_type=model_type,
        loss_profile=loss_profile,
    )

    config.result_dir = result_dir
    config.run_id = run_id
    config.run_time = time_str

    print(f"本次训练编号: 第 {config.run_id} 次")
    print(f"本次训练时间: {config.run_time}")
    print(f"实验类型: {getattr(config, 'exp', 'unknown')}")
    print(f"模型类型: {model_type}")
    print(f"loss_profile: {loss_profile}")
    print(f"训练结果保存目录: {config.result_dir}")
    print(f"模型保存/加载路径: {config.model_path}")
    print(f"预训练模型路径: {config.pretrain_model_path}")

    save_run_info(
        config=config,
        args=args,
        model_path=config.model_path,
        pretrain_model_path=config.pretrain_model_path,
    )

    # =====================
    # 7. 样本数设置
    # =====================
    max_train_samples, max_val_samples, max_test_samples = resolve_sample_limits(
        config,
        args,
    )

    print(f"max_train_samples: {max_train_samples}")
    print(f"max_val_samples: {max_val_samples}")
    print(f"max_test_samples: {max_test_samples}")
    print(f"num_epochs: {config.num_epochs}")
    print(f"batch_size: {config.batch_size}")

    # =====================
    # 8. 路径检查
    # =====================
    check_required_paths(
        config=config,
        args=args,
        pretrain_model_path=config.pretrain_model_path if not args.load_model else None,
    )

    if args.load_model and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"--load_model 指定的模型文件不存在: {config.model_path}")

    # =====================
    # 9. 数据集
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

    if args.index < 0 or args.index >= len(test_dataset):
        raise IndexError(
            f"--index {args.index} 超出测试集范围，当前 test_dataset size = {len(test_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    # 10. 模型工具
    # 注意：
    # 原 ModelTools 里 LoadModel=True 会按它自己的默认路径加载。
    # 这里统一先用 LoadModel=False 初始化，再手动指定 modeltools.model_path。
    # 这样不需要改 ModelTools.py，也能避免 exp1/exp2/exp3 checkpoint 互相覆盖。
    # =====================
    modeltools = ModelTools(
        config,
        model_type=model_type,
        loss_profile=loss_profile,
        LoadModel=False,
    )

    modeltools.model_path = config.model_path
    print(f"ModelTools 实际使用模型路径: {modeltools.model_path}")

    # =====================
    # 11. 加载模型 / 加载预训练
    # =====================
    if args.load_model:
        print(f"加载已有模型并直接测试: {modeltools.model_path}")
        modeltools.load_model(modeltools.model_path)

    elif config.pretrain_model_path is not None:
        print(f"加载预训练模型作为初始权重: {config.pretrain_model_path}")
        modeltools.load_model(config.pretrain_model_path)
        print("预训练模型加载完成，开始微调训练。")

    # =====================
    # 12. 训练
    # =====================
    if not args.load_model:
        if config.pretrain_model_path is None:
            print("不加载预训练模型，开始训练...")
        else:
            print("开始微调训练...")

        modeltools.train(train_loader, val_loader, num_epochs=config.num_epochs)

        # 训练后加载验证集最优模型
        modeltools.load_model(modeltools.model_path)

    # =====================
    # 13. 单样本预测可视化
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

    # =====================
    # 14. 测试集 MSE
    # =====================
    test_loss_mse = modeltools.validate(test_loader)
    print(f"Test Loss MSE: {test_loss_mse:.6f}")


if __name__ == "__main__":
    main()
