import argparse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='bench_scalar'
    )    
    parser.add_argument('outfile')
    parser.add_argument('-n1', '--name1')
    parser.add_argument('-d1', '--data1')
    parser.add_argument('-n2', '--name2')
    parser.add_argument('-d2', '--data2')
    parser.add_argument('-x', '--xlabel')
    args = parser.parse_args()
    
    # Load results
    data1 = np.fromfile('data/' + args.data1, dtype=np.float64)
    data2 = np.fromfile('data/' + args.data2, dtype=np.float64)
    
    # Draw results
    n = data1.shape[0]
    assert data1.shape == data2.shape
    
    plt.bar(np.arange(1, n + 1), data1, width=0.25, color='r', label=args.name1)
    plt.bar(np.arange(1, n + 1) + 0.25, data2, width=0.25, color='g', label=args.name2)
    plt.xlabel(args.xlabel)
    plt.ylabel(r'time (seconds)')
    plt.legend()
    
    plt.savefig(args.outfile)

    