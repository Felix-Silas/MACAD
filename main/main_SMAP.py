import os
import subprocess
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Run MACAD on SMAP domain pairs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    run_args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.abspath(os.path.join(project_root, '..', 'datasets', 'SMAP'))
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
        smap_ids = set(metadata.loc[metadata['spacecraft'] == 'SMAP', 'chan_id'])
        files = sorted(name for name in all_names if name in smap_ids)

    '''DESIRED_TARGETS = [
    'A-1', 'A-2', 'A-3']

    files = [f for f in files if f in DESIRED_TARGETS]'''

    print(files)

    #for src in ['A-7', 'D-7', 'E-8', 'P-2']:
    for src in ['T-2']:
        for trg in files:
            if src != trg:
                print('src: ', src, ' / target: ', trg)

                train = os.path.join(project_root, 'main', 'train.py')
                command = ['python', train, '--algo_name', 'macad', '--num_epochs', '20',
                           '--batch_size', '256', '--eval_batch_size', '256', '--learning_rate',
                           '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7',
                           '--id_src', src, '--id_trg', trg, '--experiment_folder', 'SMAP',
                           '--path_src', dataset_root, '--path_trg', dataset_root,
                           '--seed', *map(str, run_args.seeds)]
                subprocess.run(command)

                test = os.path.join(project_root, 'main', 'eval.py')
                for seed in run_args.seeds:
                    command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder',
                                'SMAP', '--id_src', src, '--id_trg', trg, '--seed', str(seed)]
                    subprocess.run(command1)



'''
['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9', 'B-1', 'D-1', 'D-11', 'D-12', 'D-13', 
'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'E-1', 'E-10', 'E-11', 'E-12', 'E-13', 'E-2', 
'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 
'G-7', 'P-1', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3']
'''
