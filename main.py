from torch.utils.data import DataLoader
from config import Config
from PIDataset import PeakInversionDataset
from ModelTools import ModelTools
from utils.visualize import visualize_prediction
import argparse


LOSS_PROFILES = [
    'base',
    'mse_grad',
    'huber_grad',
    'pinn',
    'pinn_smooth',
    'pinn_tikhonov',
    'pinn_huber',
    'pinn_full',
]


def parse_args():
    parser = argparse.ArgumentParser(description='峰值反演模型训练与评估')

    parser.add_argument(
        '--model_type',
        type=str,
        default='transformer',
        choices=['cnn', 'unet', 'transformer'],
        help='选择网络结构: cnn, unet 或 transformer',
    )

    parser.add_argument(
        '--loss_profile',
        type=str,
        default=None,
        choices=LOSS_PROFILES,
        help='选择损失组合: base, mse_grad, pinn, pinn_smooth 等；不填则使用 config.loss_profile',
    )

    # 兼容之前版本；现在真正生效的是 loss_profile
    parser.add_argument(
        '--train_type',
        type=str,
        default=None,
        choices=['supervised', 'pinn'],
        help='兼容旧参数。supervised 默认对应 base，pinn 默认对应 config.loss_profile',
    )

    parser.add_argument('--load_model', action='store_true', help='是否加载预训练模型')
    parser.add_argument('--index', type=int, default=0, help='用于预测的样本索引')

    parser.add_argument('--max_train_samples', type=int, default=None, help='覆盖 config.max_train_samples')
    parser.add_argument('--max_val_samples', type=int, default=None, help='覆盖 config.max_val_samples')
    parser.add_argument('--max_test_samples', type=int, default=None, help='覆盖 config.max_test_samples')

    return parser.parse_args()


def main():
    config = Config()
    args = parse_args()

    model_type = args.model_type

    if args.loss_profile is not None:
        loss_profile = args.loss_profile
    elif args.train_type == 'supervised':
        loss_profile = 'base'
    else:
        loss_profile = config.loss_profile

    max_train_samples = args.max_train_samples if args.max_train_samples is not None else config.max_train_samples
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else config.max_val_samples
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else config.max_test_samples

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

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    modeltools = ModelTools(
        config,
        model_type=model_type,
        loss_profile=loss_profile,
        LoadModel=args.load_model,
    )

    if not args.load_model:
        print('不加载预训练模型，开始训练...')
        modeltools.train(train_loader, val_loader, num_epochs=config.num_epochs)
        try:
            modeltools.load_model(modeltools.model_path)
        except Exception:
            pass

    gy, fx_true = test_dataset.__getitem__(args.index)
    x = test_dataset.__getx__(args.index)
    fx_pred, loss = modeltools.predict(predict_loader=predict_loader, index=args.index)

    print(f'Predicted fx shape: {fx_pred.shape}')
    print(f'Ground truth fx shape: {fx_true.shape}')
    print(f'Prediction Loss MSE: {loss:.6f}')
    print(f'Predicted fx: {fx_pred}')

    visualize_prediction(
        fx_true.squeeze().numpy(),
        fx_pred.squeeze().detach().cpu().numpy(),
        x,
    )

    test_loss_mse = modeltools.validate(test_loader)
    print(f'Test Loss MSE: {test_loss_mse:.6f}')


if __name__ == '__main__':
    main()
