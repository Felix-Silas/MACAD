import os
import subprocess
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Run MACAD on SMD domain pairs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    run_args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.abspath(os.path.join(project_root, '..', 'datasets', 'SMD'))
    files = sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(dataset_root, name, 'train.csv'))
        and os.path.isfile(os.path.join(dataset_root, name, 'test.csv'))
    )
    if not files:
        all_files = os.listdir(os.path.join(dataset_root, 'train'))
        files = sorted(file for file in all_files if file.startswith('machine-'))

        DESIRED_TARGETS = ['machine-3-1.txt', 'machine-3-11.txt', 'machine-3-2.txt', 'machine-3-3.txt', 'machine-3-4.txt', 'machine-3-5.txt', 'machine-3-6.txt', 'machine-3-7.txt', 'machine-3-8.txt', 'machine-3-9.txt']
        files = [f for f in files if f in DESIRED_TARGETS]

    print(files)
    # 'machine-1-1.txt'
    for src in ['machine-3-10.txt']:
    #for src in [ 'machine-2-3.txt', 'machine-3-7.txt', 'machine-1-5.txt']:
        src = src.replace('machine-', '')
        src = src.replace('.txt', '')
        for trg in files:
            trg = trg.replace('machine-', '')
            trg = trg.replace('.txt', '')
            
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                train = os.path.join(project_root, 'main', 'train.py')
                command = ['python', train, '--algo_name', 'macad', '--num_epochs', '20', '--batch_size', '128',
                           '--eval_batch_size', '256', '--learning_rate', '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7', '--id_src', src, '--id_trg', trg, '--experiment_folder', 'SMD',
                           '--path_src', dataset_root, '--path_trg', dataset_root,
                           '--seed', *map(str, run_args.seeds)]
                subprocess.run(command)

                test = os.path.join(project_root, 'main', 'eval.py')
                for seed in run_args.seeds:
                    command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder', 'SMD', '--id_src', src, '--id_trg', trg, '--seed', str(seed)]
                    subprocess.run(command1)
