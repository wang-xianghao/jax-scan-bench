import jax
import jax.numpy as jnp

import numpy as np

import time
import argparse

def test_one(nums, shape):
    X = jax.random.uniform(jax.random.key(0), (nums, ) + shape, dtype=jnp.float32)
    
    t_start = time.perf_counter()
    A = jax.lax.associative_scan(jnp.matmul, X)
    t_end = time.perf_counter()
    
    return t_end - t_start

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='bench_scalar'
    )    
    parser.add_argument('outfile')
    parser.add_argument('-s', '--size', type=int)
    parser.add_argument('-b', '--base', type=int)
    parser.add_argument('-p', '--power', type=int)

    args = parser.parse_args()

    # Collect bench results
    elapsed_all = np.zeros(args.power)
    
    for p in range(1, args.power + 1):
        n = args.base ** p
        elapsed_all[p - 1] = test_one(n, (args.size, args.size))
        print(f'Bench of size {n} finished in {elapsed_all[p - 1]} seconds')
        
    elapsed_all.tofile('data/' + args.outfile)