import jax
import jax.numpy as jnp

import numpy as np

import time
import argparse

def test_one(nums):
    X = jnp.linspace(0, 1, nums, dtype=jnp.float32)
    
    t_start = time.perf_counter()
    A = jax.lax.associative_scan(jnp.multiply, X)
    t_end = time.perf_counter()
    
    return t_end - t_start

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='bench_scalar'
    )    
    parser.add_argument('outfile')
    parser.add_argument('-b', '--base', type=int)
    parser.add_argument('-p', '--power', type=int)

    args = parser.parse_args()

    # Collect bench results
    elapsed_all = np.zeros(args.power)
    
    for p in range(1, args.power + 1):
        n = args.base ** p
        elapsed_all[p - 1] = test_one(n)
        print(f'Bench of size {n} finished in {elapsed_all[p - 1]} seconds')
        
    elapsed_all.tofile('data/' + args.outfile)