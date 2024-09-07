import jax
import jax.numpy as jnp
from jax import lax

import time 
import sys

if len(sys.argv) != 2:
    print('Usage: ./simpleAdd <#elements>')
    exit(0)

nums = int(sys.argv[1])

X = jnp.arange(0, nums)

print('Device: ', X.device)

t_start = time.perf_counter()
A = lax.associative_scan(jnp.add, X)
t_end = time.perf_counter()

print(f'Elapsed time: {t_end - t_start: 10.6f} seconds')