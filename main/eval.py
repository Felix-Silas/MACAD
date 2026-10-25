import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import get_logger, \
    get_dataset_type

import json

from argparse import ArgumentParser
from collections import namedtuple

from algorithms import get_algorithm
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import time
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import pickle


def resolve_run_dir(args):
    pair_dir = os.path.join(
        args.experiments_main_folder,
        args.experiment_folder,
        f"{args.id_src}-{args.id_trg}",
    )
    seeded_dir = os.path.join(pair_dir, f"seed-{args.seed}")
    # Allow evaluation of checkpoints produced before seed directories existed.
    return seeded_dir if os.path.isfile(os.path.join(seeded_dir, 'commandline_args.txt')) else pair_dir


def main(args):
    total_start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[Eval] Evaluation started at {current_time}")

    # ==================== 여기 추가: device 설정 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")

    run_dir = resolve_run_dir(args)
    with open(os.path.join(run_dir, 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)

    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    #print(saved_args)
    # configure our logger
    log = get_logger(os.path.join(run_dir, "eval_" + saved_args.log))
    
    log(f"Evaluation started at {current_time}")
    log(f"Source ID: {args.id_src} → Target ID: {args.id_trg}")

    # Some functions and variables for logging
    dataset_type = get_dataset_type(saved_args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type in {"smd", "msl", "boiler", "smap"}:
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
            log("Fixed-threshold F1 score is : %.4f " % (metrics_pred["f1"]))
            log("Fixed-threshold precision is : %.4f " % (metrics_pred["precision"]))
            log("Fixed-threshold recall is : %.4f " % (metrics_pred["recall"]))
            log("Validation threshold is : %.6f " % (metrics_pred["threshold"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

    data_load_start = time.time()
    dataset_val_src = get_dataset(saved_args, domain_type="source", split_type="val")
    data_load_time = time.time() - data_load_start
    log(f"Source validation dataset loading took {data_load_time:.2f} seconds")


    # Calculate the MACAD input dimension.
    input_channels_dim = dataset_val_src[0]['sequence'].shape[1]

    experiment_folder_path = run_dir
    
    # Get trained algorithm for After DA
    algorithm_trained = get_algorithm(saved_args, input_channels_dim=input_channels_dim)
    algorithm_trained.load_state(experiment_folder_path)
    algorithm_trained.eval()
    algorithm_trained.to(device)

    # Select the paper's validation-optimal threshold from labeled source-val.
    # Target test remains unopened until both model and threshold are fixed.
    loader_val_src = DataLoader(
        dataset_val_src, batch_size=eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    with torch.no_grad():
        for sample_batched in loader_val_src:
            for key, value in sample_batched.items():
                if isinstance(value, torch.Tensor):
                    sample_batched[key] = value.to(device, non_blocking=True)
            algorithm_trained.predict_src(sample_batched)

    source_val_metrics = algorithm_trained.pred_meter_val_src.get_metrics()
    threshold_trg = float(source_val_metrics["best_thr"])
    log(f"Threshold selected on source-val: {threshold_trg:.6f}")
    algorithm_trained.init_pred_meters_val()

    # Target test is loaded only after model and threshold are fully fixed.
    dataset_test_trg = get_dataset(saved_args, domain_type="target", split_type="test")
    
    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                                     shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    with torch.no_grad():
        for sample_batched in dataloader_test_trg:
            for key, value in sample_batched.items():
                if isinstance(value, torch.Tensor):
                    sample_batched[key] = value.to(device, non_blocking=True)
            algorithm_trained.predict_trg(sample_batched)

    # even though the name is "pred_meter_val_trg", in this script it saves test results
    y_test_trg = np.array(algorithm_trained.pred_meter_val_trg.target_list)
    y_pred_trg = np.array(algorithm_trained.pred_meter_val_trg.output_list)
    pred_trg_df = pd.DataFrame({"y": y_test_trg, "y_pred": y_pred_trg})
    df_save_path_trg = os.path.join(run_dir, "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)

    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")

    metrics_pred_test_trg = algorithm_trained.pred_meter_val_trg.get_metrics_at_threshold(
        threshold_trg
    )

    log_scores(saved_args, dataset_type, metrics_pred_test_trg)
    df_trg = pd.DataFrame.from_dict(metrics_pred_test_trg, orient='index')
    df_trg = df_trg.T
    df_trg.insert(0, 'src_id', args.id_src)
    df_trg.insert(1, 'trg_id', args.id_trg)
    df_trg.insert(2, 'seed', args.seed)

    fname = args.output_csv or f"{dataset_type}_{args.id_src}_seed-{args.seed}.csv"
    output_dir = os.path.dirname(os.path.abspath(fname))
    os.makedirs(output_dir, exist_ok=True)

    # Upsert instead of blindly appending so interrupted/retried experiment
    # batches cannot leave duplicate source-target rows in the result table.
    if os.path.isfile(fname):
        previous = pd.read_csv(fname)
        same_run = ((previous['src_id'].astype(str) == str(args.id_src)) &
                    (previous['trg_id'].astype(str) == str(args.id_trg)))
        if 'seed' in previous:
            same_run &= previous['seed'].astype(int) == args.seed
        previous = previous.loc[~same_run]
        df_trg = pd.concat([previous, df_trg], ignore_index=True)

    df_trg.to_csv(fname, index=False)

    '''# ─────────────────────────────────────────────────────────────
    # t-SNE 시각화 — After DA만
    # ─────────────────────────────────────────────────────────────
    log("Generating t-SNE visualization (After DA only)...")

    font_path = "/mnt/c/Windows/Fonts/times.ttf"
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    def extract_embeddings_and_labels(algorithm, dataloader, device):
        algorithm.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)
                emb = algorithm.get_embedding(batch)
                embeddings.append(emb.cpu().numpy())
                if 'label' in batch:
                    batch_labels = batch['label'].cpu().numpy().flatten().astype(int)
                elif 'y' in batch:
                    batch_labels = batch['y'].cpu().numpy().flatten().astype(int)
                else:
                    log("Warning: No 'label' or 'y' key → assuming all normal (0)")
                    batch_labels = np.zeros(emb.shape[0], dtype=int)
                labels.append(batch_labels)
        if not embeddings:
            log("Warning: No embeddings collected.")
            return np.array([]), np.array([])
        log(f"Extracted {np.concatenate(embeddings).shape[0]} samples")
        return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

    # After DA embeddings 추출
    log("→ After DA embeddings")
    emb_src_after, labels_src_after = extract_embeddings_and_labels(algorithm_trained, dataloader_test_src, device)
    emb_trg_after, labels_trg_after = extract_embeddings_and_labels(algorithm_trained, dataloader_test_trg, device)

    embeddings_after = np.concatenate([emb_src_after, emb_trg_after], axis=0)
    labels_after = np.concatenate([labels_src_after, labels_trg_after], axis=0)
    domain_after = np.concatenate([np.zeros(len(emb_src_after)), np.ones(len(emb_trg_after))])

    if len(embeddings_after) == 0:
        log("No embeddings → skipping t-SNE")
    else:
        log("Fitting t-SNE for After DA...")
        tsne_after = TSNE(
            n_components=2,
            perplexity=100,
            n_iter=10000,
            init='pca',
            learning_rate='auto',
            random_state=42
        ).fit_transform(embeddings_after)

        scaler_after = MinMaxScaler(feature_range=(0, 1))
        tsne_after_norm = scaler_after.fit_transform(tsne_after)

        # anomaly filtering
        normal_points = tsne_after_norm[labels_after == 0]
        center = normal_points.mean(axis=0) if len(normal_points) > 0 else np.array([0.5, 0.5])
        normal_dist = np.linalg.norm(normal_points - center, axis=1)
        threshold = normal_dist.mean() + 2 * normal_dist.std() if len(normal_dist) > 0 else 0.5

        anomaly_indices = np.where(labels_after == 1)[0]
        anomaly_points = tsne_after_norm[anomaly_indices]
        anomaly_dist = np.linalg.norm(anomaly_points - center, axis=1)
        keep_anomaly_mask = anomaly_dist > threshold
        anomaly_keep_mask = np.zeros(len(labels_after), dtype=bool)
        anomaly_keep_mask[anomaly_indices[keep_anomaly_mask]] = True   # 수정: keep_anomaly_mask를 사용

        log(f"Filtered anomaly points: {len(anomaly_indices) - np.sum(keep_anomaly_mask)} removed")

        # ─────────────────────────────────────────────────────────────
        # After DA t-SNE 데이터 pickle 저장
        # ─────────────────────────────────────────────────────────────
        tsne_data = {
            "tsne_coords": tsne_after_norm,
            "domain": domain_after.astype(np.int32),
            "labels": labels_after.astype(np.int32),
            "anomaly_keep_mask": anomaly_keep_mask,
            "normal_center": center.tolist(),
            "anomaly_threshold": float(threshold),
            "src_id": args.id_src,
            "trg_id": args.id_trg,
            "n_src": len(emb_src_after),
            "n_trg": len(emb_trg_after),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_dir = os.path.join(args.experiments_main_folder, "adaptation_abnormal", "pkl")
        os.makedirs(save_dir, exist_ok=True)

        pkl_filename = f"tsne_after_{args.id_src}_to_{args.id_trg}.pkl"
        pkl_path = os.path.join(save_dir, pkl_filename)

        with open(pkl_path, 'wb') as f:
            pickle.dump(tsne_data, f)

        log(f"After DA t-SNE data saved: {pkl_path}")

        # ─────────────────────────────────────────────────────────────
        # After DA만 플롯
        # ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 10))

        ax.set_title(f"After DA ({args.id_src} → {args.id_trg})", fontsize=28, fontweight='bold', pad=25)

        # source normal
        mask = (domain_after == 0) & (labels_after == 0)
        ax.scatter(tsne_after_norm[mask, 0], tsne_after_norm[mask, 1],
                   c='#6495ed', marker='o', s=60, alpha=0.6,
                   label=f'{args.id_src} Normal')

        # source anomaly (필터링 후)
        mask = (domain_after == 0) & (labels_after == 1) & anomaly_keep_mask
        ax.scatter(tsne_after_norm[mask, 0], tsne_after_norm[mask, 1],
                   c='#00008b', marker='^', s=80, alpha=0.9,
                   label=f'{args.id_src} Anomaly')

        # target normal
        mask = (domain_after == 1) & (labels_after == 0)
        ax.scatter(tsne_after_norm[mask, 0], tsne_after_norm[mask, 1],
                   c='#ff6347', marker='o', s=60, alpha=0.6,
                   label=f'{args.id_trg} Normal')

        # target anomaly (필터링 후)
        mask = (domain_after == 1) & (labels_after == 1) & anomaly_keep_mask
        ax.scatter(tsne_after_norm[mask, 0], tsne_after_norm[mask, 1],
                   c='#8b0000', marker='^', s=80, alpha=0.9,
                   label=f'{args.id_trg} Anomaly')

        ax.grid(True, linestyle='--', alpha=0.5)

        ax.legend(title='Domain & Label', loc='upper right', fontsize=14, title_fontsize=16,
                  markerscale=1.5, frameon=True)

        save_dir_png = os.path.join(args.experiments_main_folder, "adaptation_abnormal")
        os.makedirs(save_dir_png, exist_ok=True)
        save_filename = f"tsne_after_{args.id_src}_to_{args.id_trg}.png"
        save_path = os.path.join(save_dir_png, save_filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        log(f"Saved After DA t-SNE plot: {save_path}")'''

    # 전체 시간 로그
    total_time = time.time() - total_start_time
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log(f"\nEvaluation finished at {end_time}")
    log(f"Total evaluation time: {total_time:.2f} seconds")
    log("="*60)


if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('--id_src', type=str, default='1-1')
    parser.add_argument('--id_trg', type=str, default='1-5')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed identifying the trained run (default: 42).')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Optional result CSV path (rows are upserted by src_id/trg_id).')

    args = parser.parse_args()

    main(args)
