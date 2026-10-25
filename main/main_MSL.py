import os
import subprocess
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Run MACAD on MSL domain pairs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    run_args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.abspath(os.path.join(project_root, '..', 'datasets', 'MSL'))
    # Prefer the shared Boiler-style <domain>/train.csv,test.csv layout.
    files = sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(dataset_root, name, 'train.csv'))
        and os.path.isfile(os.path.join(dataset_root, name, 'test.csv'))
    )
    if not files:
        # Backward-compatible support for the original NASA .npy layout.
        all_files = os.listdir(os.path.join(dataset_root, 'train'))
        all_names = [name[:-4] for name in all_files if name.endswith('.npy')]
        metadata = pd.read_csv(os.path.join(dataset_root, 'labeled_anomalies.csv'),
                               usecols=['chan_id', 'spacecraft'])
        msl_ids = set(metadata.loc[metadata['spacecraft'] == 'MSL', 'chan_id'])
        files = sorted(name for name in all_names if name in msl_ids)
    print(files)

    for src in ['F-5']:
        for trg in files:
            if src != trg:
                print('src: ', src, ' / target: ', trg)

                train = os.path.join(project_root, 'main', 'train.py')
                command = ['python', train, '--algo_name', 'macad', '--num_epochs', '20',
                           '--batch_size', '256', '--eval_batch_size', '256', '--learning_rate',
                           '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7',
                           '--id_src', src, '--id_trg', trg, '--experiment_folder', 'MSL',
                           '--path_src', dataset_root, '--path_trg', dataset_root,
                           '--seed', *map(str, run_args.seeds)]
                subprocess.run(command)

                test = os.path.join(project_root, 'main', 'eval.py')
                for seed in run_args.seeds:
                    command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder',
                                'MSL', '--id_src', src, '--id_trg', trg, '--seed', str(seed)]
                    subprocess.run(command1)
