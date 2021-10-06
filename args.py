import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_dir', default='ModelNet10/ModelNet10', type=str, help='dataset directory')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--saved_model', default='./checkpoints/', type=str, help='checkpoints dir')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    args = parser.parse_args()
    assert args.root_dir is not None
    print(' '.join(sys.argv))
    print(args)

    return args
