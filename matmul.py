import jax
import jax.numpy as jnp
from jax import lax

import time 
import sys

if len(sys.argv) != 3:
    print('Usage: ./matmul <#matrices> <#width>')
    exit(0)

nums = int(sys.argv[1])
width = int(sys.argv[2])

X = jax.random.uniform(jax.random.key(0), (nums, width, width), dtype=jnp.float32)

print('Device: ', X.device)

t_start = time.perf_counter()
A = lax.associative_scan(jnp.matmul, X)
t_end = time.perf_counter()

print(f'Elapsed time: {t_end - t_start: 10.6f} seconds')