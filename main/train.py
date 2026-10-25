import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
import torch
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import ProgressMeter, get_logger, get_dataset_type
import json
import argparse
from argparse import ArgumentParser
from algorithms import get_algorithm
import time


def get_run_dir(args):
    return os.path.join(
        args.experiments_main_folder,
        args.experiment_folder,
        f"{args.id_src}-{args.id_trg}",
        f"seed-{args.seed}",
    )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 랜덤 시드 고정하여 재현성 보장
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)
    ##
    total_start_time = time.time()

    # First configure our logger
    run_dir = get_run_dir(args)
    log = get_logger(os.path.join(run_dir, args.log))

    log(f"Main function started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Some functions and variables for logging
    dataset_type = get_dataset_type(args)

    def log_scores(args, dataset_type, metrics_pred, prefix = ''):
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))

        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))

        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))

        elif dataset_type == "smap":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))

        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size

    # LOAD SOURCE and TARGET datasets
    data_load_start = time.time()

    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    dataset_val_src = get_dataset(args, domain_type="source", split_type="val")
    dataset_trg = get_dataset(args, domain_type="target", split_type="train")


    data_load_time = time.time() - data_load_start
    log(f"Dataset loading took {data_load_time:.2f} seconds")


    dataloader_start = time.time()

    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    dataloader_time = time.time() - dataloader_start
    log(f"DataLoader creation took {dataloader_time:.2f} seconds")


    # Calculate the MACAD input dimension.
    input_channels_dim = dataset_src[0]['sequence'].shape[1]

    algo_init_start = time.time()

    # Get our algorithm
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim)

    center_loader = DataLoader(dataset_src, batch_size=eval_batch_size,
                               shuffle=False, num_workers=2, pin_memory=True,
                               drop_last=False)
    algorithm.initialize_center(center_loader)
    log("DeepSVDD center initialized from source-train normal features and fixed")

    algo_init_time = time.time() - algo_init_start
    log(f"Algorithm initialization took {algo_init_time:.2f} seconds")

    experiment_folder_path = run_dir
    
    # Initialize progress metrics before training
    global_step = 0
    best_val_score = float("-inf")

    for i in range(args.num_epochs):
        count_step = 0
        epoch_start = time.time()
        log(f"Starting epoch {i+1} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        dataloader_iterator = iter(dataloader_trg)

        for i_batch, sample_batched_src in enumerate(dataloader_src):
                batch_start = time.time()
                sample_batched_src = sample_batched_src
                for key, value in sample_batched_src.items():
                    if isinstance(value, torch.Tensor):
                        sample_batched_src[key] = value.to(device, non_blocking=True)
                # Current model does not support smaller batches than batch_size (due to queue ptr)
                if len(sample_batched_src['sequence']) != batch_size:
                    continue

                try:
                    sample_batched_trg = next(dataloader_iterator)
                except StopIteration:
                    dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                                shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
                    dataloader_iterator = iter(dataloader_trg)
                    sample_batched_trg = next(dataloader_iterator)

                for key, value in sample_batched_trg.items():
                    if isinstance(value, torch.Tensor):
                        sample_batched_trg[key] = value.to(device, non_blocking=True)

                # Current model does not support smaller batches than batch_size (due to queue ptr)
                if len(sample_batched_trg['sequence']) != batch_size:
                    continue
                
                step_start = time.time()
                # Training step of algorithm
                algorithm.step(
                    sample_batched_src, sample_batched_trg,
                    count_step=global_step, epoch=i,
                )
                global_step += 1
                step_time = time.time() - step_start
                print(f"Training step {count_step + 1} took {step_time:.2f} seconds")
        

                count_step += 1
                if count_step % len(dataloader_src) == 0:
                    progress = ProgressMeter(
                        len(dataloader_src),
                        algorithm.return_metrics(),
                        prefix="Epoch: [{}]".format(i+1))

                    log(progress.display(i_batch + 1, is_logged=True))

                    metrics = algorithm.return_metrics()
                    train_metrics = {
                        "train/loss_sup": metrics[0].avg,  # L_Src_Sup
                        "train/loss_sup_con": metrics[1].avg,  # L_Src_Sup_Con
                        "train/loss_sup_tem": metrics[2].avg,  # L_Src_Sup_Tem
                        "train/loss_sup_tran": metrics[3].avg,  # L_Src_Sup_Tran
                        "train/loss_trg_ccl": metrics[4].avg,
                        "train/loss_trg_con": metrics[5].avg,
                        "train/loss_trg_tem": metrics[6].avg,
                        "train/loss_trg_tran": metrics[7].avg,
                        "train/loss_disc": metrics[8].avg,  # Loss Disc
                        "train/loss_pred": metrics[9].avg,  # Loss Pred
                        "train/loss_total": metrics[10].avg  # Loss TOTAL
                    }

                    # Refresh the saved metrics for algorithm
                    algorithm.init_metrics()

                    # Refresh the validation meters of algorithm
                    algorithm.init_pred_meters_val()

                    # turn algorithm into eval mode
                    val_start = time.time()
                    algorithm.eval()

                    with torch.no_grad():
                        for sample_batched_val_src in dataloader_val_src:
                            for key, value in sample_batched_val_src.items():
                                if isinstance(value, torch.Tensor):
                                    sample_batched_val_src[key] = value.to(
                                        device, non_blocking=True
                                    )
                            algorithm.predict_src(sample_batched_val_src)

                    metrics_pred_val_src = algorithm.pred_meter_val_src.get_metrics()

                    log("VALIDATION RESULTS")

                    log("VALIDATION SOURCE PREDICTIONS")
                    log_scores(args, dataset_type, metrics_pred_val_src)

                    # Checkpoint selection uses only the labeled source-val
                    # partition. No target-test value or label is available here.
                    cur_val_score = metrics_pred_val_src["best_f1"]
                    if cur_val_score > best_val_score:
                        algorithm.save_state(experiment_folder_path)
                        best_val_score = cur_val_score
                    # turn algorithm into training mode
                    algorithm.train()

                    # Refresh the saved metrics for algorithm
                    algorithm.init_metrics()
                    
                    val_time = time.time() - val_start
                    log(f"Validation for epoch {i+1} took {val_time:.2f} seconds")
                    

                batch_time = time.time() - batch_start
                print(f"Batch {i_batch} took {batch_time:.2f} seconds")

        epoch_time = time.time() - epoch_start
        log(f"Epoch {i+1} took {epoch_time:.2f} seconds")

    total_time = time.time() - total_start_time
    log(f"Total execution time: {total_time:.2f} seconds")
    log(f"Main function ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ===== 추가: 학습 끝! weight 그래프 저장 =====
    algorithm.eval()
    plot_save_dir = os.path.join(run_dir, "plots")
    algorithm.save_weight_realtime_plot(
        save_dir=plot_save_dir,
        exp_name=f"{args.id_src}_to_{args.id_trg}_seed_{args.seed}",
    )
    print(f"\nAll training finished! Weight evolution plot saved in: {plot_save_dir}")

# parse command-line arguments and execute the main method
if __name__ == '__main__':


    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='macad')

    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)  # 2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=256)  # 2048)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_channels_TCN', type=str, default='128-256-512')
    parser.add_argument('--kernel_size_TCN', type=int, default=7)
    parser.add_argument('--dilation_factor_TCN', type=int, default=3)
    parser.add_argument('--stride_TCN', type=int, default=1)  # All TCN models
    parser.add_argument('--adaptive_temperature', type=float, default=1.0)
    parser.add_argument('--window_size', type=int, default=100)

    # MACAD objective weights
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.3) # alpha
    parser.add_argument('--weight_loss_trg_ccl', type=float, default=0.3) # beta
    parser.add_argument('--weight_loss_disc', type=float, default=0.5) # lambda
    parser.add_argument('--weight_loss_pred', type=float, default=1.0) # delta
    parser.add_argument('--weight_loss_entropy', type=float, default=0.3) # gamma

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='smd')

    parser.add_argument('--path_src', type=str, default='../datasets/MSL') #../datasets/Boiler/   ../datasets/MSL ../datasets/SMD/test
    parser.add_argument('--path_trg', type=str, default='../datasets/MSL') #../datasets/SMD/test 
    parser.add_argument('--id_src', type=str, default='1-5')
    parser.add_argument('--id_trg', type=str, default='1-1')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument(
        '--seed', '--seeds', dest='seeds', type=int, nargs='+', default=[42],
        help='One or more random seeds (default: 42).',
    )

    parsed_args = parser.parse_args()

    # Preserve order while avoiding accidental duplicate runs.
    for seed in dict.fromkeys(parsed_args.seeds):
        run_values = vars(parsed_args).copy()
        run_values.pop('seeds')
        run_values['seed'] = seed
        args = argparse.Namespace(**run_values)
        run_dir = get_run_dir(args)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        main(args)
