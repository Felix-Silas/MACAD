import os
import subprocess
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Run MACAD on Boiler domain pairs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    run_args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    boiler_root = os.path.abspath(os.path.join(project_root, '..', 'datasets', 'Boiler'))
    all_items = os.listdir(boiler_root)

    # 폴더명만 추출 (1, 2, 3)
    files = [item for item in all_items if os.path.isdir(os.path.join(boiler_root, item)) and item != 'cache']
    
    files = sorted(files)
    print(files)

    skip_pairs = {
        ('1', '2'),
        #('Boiler_#01', 'Boiler_#03'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
    }

    for src in files:
        for trg in files:
            if src != trg:
                if (src, trg) in skip_pairs:
                    continue
                
                print('src: ', src, ' / target: ', trg)

                train = os.path.join(project_root, 'main', 'train.py')
                command = ['python', train, '--algo_name', 'macad', '--num_epochs', '20', '--batch_size', '512',
                           '--eval_batch_size', '512', '--learning_rate', '1e-4', '--dropout', '0.1', '--weight_decay', '1e-4', '--num_channels_TCN', '128-256-512',
                           '--dilation_factor_TCN', '3', '--kernel_size_TCN', '7', '--id_src', src, '--id_trg', trg,
                           '--experiment_folder', 'boiler', '--path_src', boiler_root, '--path_trg', boiler_root]
                command.extend(['--seed', *map(str, run_args.seeds)])
                subprocess.run(command)

                test = os.path.join(project_root, 'main', 'eval.py')
                for seed in run_args.seeds:
                    command1 = ['python', test, '--experiments_main_folder', 'results', '--experiment_folder', 'boiler', '--id_src', src, '--id_trg', trg, '--seed', str(seed)]
                    subprocess.run(command1)
