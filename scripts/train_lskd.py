import _bootstrap
import argparse
from training.runner import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lskd/resnet32x4_resnet8x4.yaml')
    parser.add_argument('--experiment-id')
    args = parser.parse_args()
    run(args.config, expected_method='lskd', experiment_id=args.experiment_id)