#  Copyright 2018 Algolux Inc. All Rights Reserved.
import os
import argparse
import train
import eval
import eval_binned


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Run train or eval scripts for Gated2Depth')

    parser.add_argument("--base_dir", help="Path to dataset", required=True)

    parser.add_argument("--train_files_path", help="Path to file with train file names", required=False)

    parser.add_argument("--eval_files_path",
                        help="Path to file with validation/evaluation file names. Required if running both in train and eval mode",
                        required=True)

    parser.add_argument("--data_type", choices=['real', 'synthetic'], help="[real|synthetic].", default='real',
                        required=True)

    parser.add_argument("--dataset", choices=['g2d', 'stf'], help="[g2d|stf].", default='g2d',
                        required=True)

    parser.add_argument("--results_dir", help="Path to results directory (train or eval)",
                        default='gated2depth_results', required=False)

    parser.add_argument("--model_dir", help="Path to model directory",
                        default='gated2depth model', required=False)

    parser.add_argument("--exported_disc_path", help="Path to exported discriminator. Used to train "
                                                     "a generator with a pre-trained discriminator",
                        default=None, required=False)

    parser.add_argument("--mode", choices=['train', 'eval'], help="[train/eval]",
                        default='train', required=False)

    parser.add_argument('--use_multiscale', help='Use multiscale loss function',
                        action='store_true', required=False)

    parser.add_argument('--smooth_weight', type=float, help='Smoothing loss weight',
                        default=0.5, required=False)

    parser.add_argument('--adv_weight', type=float, help='Adversarial loss weight',
                        default=0., required=False)

    parser.add_argument('--lrate', type=float, help='Learning rate',
                        default=0.0001, required=False)

    parser.add_argument('--min_distance', type=float, help='minimum distance',
                        default=3., required=False)

    parser.add_argument('--max_distance', type=float, help='maximum distance',
                        default=150., required=False)

    parser.add_argument('--use_3dconv', help='Use 3D convolutions architecture',
                        action='store_true', required=False)

    parser.add_argument('--use_filtered_lidar', help='Use filtered lidar data for training and evaluation',
                        action='store_true', required=False)

    parser.add_argument('--gpu', dest='gpu', help='GPU id', default='0', required=False)

    parser.add_argument('--num_epochs', type=int, dest='num_epochs',
                        help='Number of training epochs', default=2)

    parser.add_argument('--show_result', help='Show result image during evaluation',
                        action='store_true', required=False)
    parser.add_argument('--binned_metric', help='Calculate binned metric',
                        action='store_true', required=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    with open(os.path.join(args.results_dir, args.mode + '_parameters.txt'), 'w') as f:
        f.write(str(args))

    if args.mode == 'train':
        with open(args.train_files_path, 'r') as f:
            train_file_names = [line.strip() for line in f.readlines()]
        with open(args.eval_files_path, 'r') as f:
            eval_file_names = [line.strip() for line in f.readlines()]

        train.run(args.results_dir, args.model_dir, args.base_dir, train_file_names, eval_file_names, args.num_epochs,
                  args.data_type,
                  use_multi_scale=args.use_multiscale, exported_disc_path=args.exported_disc_path,
                  use_3dconv=args.use_3dconv, smooth_weight=args.smooth_weight,
                  adv_weight=args.adv_weight, lrate=args.lrate, min_distance=args.min_distance,
                  max_distance=args.max_distance, dataset=args.dataset, use_filtered_lidar=args.use_filtered_lidar)
    else:
        with open(args.eval_files_path, 'r') as f:
            eval_file_names = [line.strip() for line in f.readlines()]
        eval.run(args.results_dir, args.model_dir, args.base_dir, eval_file_names, args.data_type,
             args.exported_disc_path, use_3dconv=args.use_3dconv,
             compute_metrics=False, min_distance=args.min_distance, max_distance=args.max_distance, show_result=args.show_result, dataset=args.dataset, use_filtered_lidar=args.use_filtered_lidar, binned_metric=args.binned_metric)
