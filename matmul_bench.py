import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import time 
import sys

def test_one(nums, width):
    X = jax.random.uniform(jax.random.key(0), (nums, width, width), dtype=jnp.float32)

    t_start = time.perf_counter()
    A = lax.associative_scan(jnp.matmul, X)
    t_end = time.perf_counter()
    return t_end - t_start


if len(sys.argv) != 4:
    print('Usage: ./matmul_bench <nums_powers> <width_powers> <figure_path>')
    exit(0)

fig_path = sys.argv[3]

nums_powers = int(sys.argv[1])
width_powers = int(sys.argv[2])
elapsed_all = np.zeros((width_powers + 1, nums_powers + 1))

for n in range(0, nums_powers + 1):
    for w in range(0, width_powers + 1):
        nums, width = 2 ** n, 2 ** w
        elapsed_all[w, n] = test_one(nums, width)

hm = sns.heatmap(data=elapsed_all, annot=True) 
plt.xlabel('Number of Matrices (log2)')
plt.ylabel('Matrix width (log2)')
plt.savefig(fig_path)