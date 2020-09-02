from future import standard_library

standard_library.install_aliases()
import argparse
import os

import matplotlib

matplotlib.use('Agg')  # Needed to run without X-server
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--file', action='append', dest='files',
                        default=[], type=str,
                        help='specify paths of scores.txt')
    parser.add_argument('--label', action='append', dest='labels',
                        default=[], type=str,
                        help='specify labels for scores.txt files')
    parser.add_argument('--mean', action='store_true', default=False)
    args = parser.parse_args()

    assert len(args.files) > 0
    assert len(args.labels) == len(args.files)

    for fpath, label in zip(args.files, args.labels):
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, 'scores.txt')
        assert os.path.exists(fpath)
        scores = pd.read_csv(fpath, delimiter='\t')
        if args.mean:
            plt.plot(scores['steps'], scores['mean'], label=label)
        else:
            plt.plot(scores['steps'], scores['median'], label=label)

    plt.xlabel('steps')
    if args.mean:
        plt.ylabel('mean')
    else:
        plt.ylabel('median')
    plt.legend(loc='best')
    if args.title:
        plt.title(args.title)
    if args.mean:
        fig_fname = args.files[0] + args.title + 'mean' + '.png'
    else:
        fig_fname = args.files[0] + args.title + 'median' + '.png'
    plt.savefig(fig_fname)
    print('Saved a figure as {}'.format(fig_fname))


if __name__ == '__main__':
    main()
