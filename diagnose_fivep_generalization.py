from pathlib import Path
import argparse, json, csv
from types import SimpleNamespace
import numpy as np
import torch
import train_fivep_compare as base


def metrics_from_physical(true, pred):
    true = np.asarray(true, np.float64)
    pred = np.asarray(pred, np.float64)
    abs_err = np.abs(pred - true)
    gf = np.maximum(pred[:,4]/np.maximum(true[:,4],1e-300), true[:,4]/np.maximum(pred[:,4],1e-300))
    pa1 = abs_err[:,0] <= base.RECOVERY['a1_abs']
    pm = abs_err[:,3] <= base.RECOVERY['m_abs']
    pg = gf <= base.RECOVERY['gamma_factor']
    joint = pa1 & pm & pg
    out = {
        'n': int(len(true)),
        'R_a1': float(pa1.mean()),
        'R_m': float(pm.mean()),
        'R_gamma': float(pg.mean()),
        'R_joint': float(joint.mean()),
        'a1_mae': float(abs_err[:,0].mean()),
        'a2_mae': float(abs_err[:,1].mean()),
        'a3_mae': float(abs_err[:,2].mean()),
        'm_mae': float(abs_err[:,3].mean()),
        'gamma_mae': float(abs_err[:,4].mean()),
        'gamma_factor_p90': float(np.quantile(gf,0.90)),
    }
    return out


def aggregate_state_metrics(ds, pred_norm, norm):
    pred = base.pred_norm_to_physical(pred_norm, norm)
    ids = np.asarray(ds.ids)
    true = np.asarray(ds.parameters, np.float64)
    uniq = np.unique(ids)
    true_state, pred_state = [], []
    for sid in uniq:
        mask = ids == sid
        true_state.append(true[mask][0])
        pred_state.append(pred[mask].mean(axis=0))
    return metrics_from_physical(np.asarray(true_state), np.asarray(pred_state))


def main():
    ap = argparse.ArgumentParser(description='Diagnose Train-vs-Val generalization from existing five-parameter checkpoints. NO TEST access.')
    ap.add_argument('--data-dir', default=r'.\fivep_train_val_v1\data')
    ap.add_argument('--runs-dir', default=r'.\runs_fivep_compare_v1')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output', default=r'.\fivep_generalization_diagnostic.csv')
    args = ap.parse_args()

    root = Path(args.data_dir)
    runs = Path(args.runs_dir)
    manifest_path = runs / 'run_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    norm = base.load_norm(root/'normalization_train_only.npz')
    train_ds = base.FivePDataset(root/'noise_0p2pct'/'train.npz', norm)
    val_ds = base.FivePDataset(root/'noise_0p2pct'/'val.npz', norm)
    base.verify_train_val(train_ds, val_ds)
    q2 = np.load(root/'q2.npy', allow_pickle=False) if (root/'q2.npy').exists() else None
    coord, coord_mode = base.make_coordinate_channels(q2, train_ds.x.shape[1])
    device = base.choose_device(args.device)
    amp_enabled = device.type == 'cuda'

    cfg = SimpleNamespace(
        tcn_channels=manifest['tcn']['channels'],
        kernel_size=manifest['tcn']['kernel'],
        tcn_dropout=manifest['tcn']['dropout'],
        pool_bins=manifest['tcn']['pool_bins'],
        mlp_hidden=manifest['mlp']['hidden'],
        mlp_dropout=manifest['mlp']['dropout'],
    )
    weights = torch.tensor(manifest.get('target_weights',[1,1,1,1,1]), dtype=torch.float32, device=device)
    train_x = torch.from_numpy(train_ds.x).to(device)
    train_y = torch.from_numpy(train_ds.y).to(device)
    val_x = torch.from_numpy(val_ds.x).to(device)
    val_y = torch.from_numpy(val_ds.y).to(device)

    rows = []
    for model_name in manifest['models']:
        for seed in manifest['seeds']:
            ckpt_path = runs/model_name/f'seed_{seed}'/'best.pt'
            if not ckpt_path.exists():
                print(f'SKIP missing: {ckpt_path}')
                continue
            model = base.build_model(model_name, train_ds.x.shape[1], coord, cfg).to(device)
            ckpt = base.safe_torch_load(ckpt_path, device)
            model.load_state_dict(ckpt['model_state'])
            tr_loss, tr_pred = base.eval_loss_pred(model, train_x, train_y, amp_enabled, 256, weights)
            va_loss, va_pred = base.eval_loss_pred(model, val_x, val_y, amp_enabled, 256, weights)
            tr_m, _ = base.compute_metrics(train_ds, tr_pred, norm)
            va_m, _ = base.compute_metrics(val_ds, va_pred, norm)
            tr_state_m = aggregate_state_metrics(train_ds, tr_pred, norm)
            row = {
                'model': model_name, 'seed': seed, 'best_epoch': ckpt.get('best_epoch',''),
                'train_loss': tr_loss, 'val_loss': va_loss,
                'train_R_a1': tr_m['R_a1'], 'train_R_m': tr_m['R_m'], 'train_R_gamma': tr_m['R_gamma'], 'train_R_joint': tr_m['R_joint'],
                'train_stateavg_R_a1': tr_state_m['R_a1'], 'train_stateavg_R_m': tr_state_m['R_m'], 'train_stateavg_R_gamma': tr_state_m['R_gamma'], 'train_stateavg_R_joint': tr_state_m['R_joint'],
                'val_R_a1': va_m['R_a1'], 'val_R_m': va_m['R_m'], 'val_R_gamma': va_m['R_gamma'], 'val_R_joint': va_m['R_joint'],
                'train_a1_mae': tr_m['a1_mae'], 'val_a1_mae': va_m['a1_mae'],
                'train_m_mae': tr_m['m_mae'], 'val_m_mae': va_m['m_mae'],
                'train_gamma_factor_p90': tr_m['gamma_factor_p90'], 'val_gamma_factor_p90': va_m['gamma_factor_p90'],
            }
            rows.append(row)
            print(f"[{model_name} {seed}] Train R=({100*tr_m['R_a1']:.1f},{100*tr_m['R_m']:.1f},{100*tr_m['R_gamma']:.1f}) joint={100*tr_m['R_joint']:.1f}% | "
                  f"Train-stateavg joint={100*tr_state_m['R_joint']:.1f}% | Val R=({100*va_m['R_a1']:.1f},{100*va_m['R_m']:.1f},{100*va_m['R_gamma']:.1f}) joint={100*va_m['R_joint']:.1f}%")
            del model
            if device.type == 'cuda': torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError('No checkpoints found. Check --runs-dir.')
    out = Path(args.output)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {out.resolve()}')
    print('NO TEST FILE WAS OPENED.')

if __name__ == '__main__':
    main()
