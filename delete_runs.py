import torch
import os
import shutil


RUNS_DIR = '/home/ubuntu/transfer-learning-conv-ai/runs'


def is_mini(dir_path):
    data = torch.load(os.path.join(dir_path, 'model_training_args.bin'))
    return data.dataset_cache == 'dataset_cache_ed_mini'


def has_checkpoints(dir_path):
    for file in sorted(os.listdir(dir_path)):
        if 'checkpoint' in file:
            return True
        if 'best_' in file:
            return True
    return False


def main():
    to_del = []
    for dir_name in sorted(os.listdir(RUNS_DIR)):
        dir_path = os.path.join(RUNS_DIR, dir_name)
        if is_mini(dir_path):
            to_del.append((dir_path, 'dataset_cache_ed_mini'))
        if not has_checkpoints(dir_path):
            to_del.append((dir_path, 'has no checkpoints'))

    print('\nWill delete these {} runs:'.format(len(to_del)))
    for (dir_path, reason) in to_del:
        print('==================')
        print(dir_path)
        print('Reason: {}'.format(reason))
        print('\nArgs:')
        data = torch.load(os.path.join(dir_path, 'model_training_args.bin'))
        print(data)
        print('\nFiles:')
        for file in sorted(os.listdir(dir_path)):
            print(file)
    print('==================')

    print('REMEMBER: ARE YOU CURRENTLY RUNNING A JOB? DON\'T DELETE THE DIR OF THE CURRENT RUN')
    user_input = input('\nDelete these {} runs? Type y/n:'.format(len(to_del)))
    if user_input == 'y':
        print('deleting...')
        for (dir_path, _) in to_del:
            shutil.rmtree(dir_path)


if __name__=="__main__":
    main()
