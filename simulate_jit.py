#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:08:45 2026

@author: usuario
"""

from os.path import join
import sys
import time

import numpy as np
from numba import njit


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask


@njit(cache=True, fastmath=True)
def jacobi_jit(u, interior_mask, max_iter, atol):
    """Hand-rolled Jacobi. Single pass per iteration, no temporaries."""
    H, W = u.shape  # 514, 514

    # Working copy so we don't mutate the caller's array (matches reference)
    u = u.copy()
    u_new = np.empty_like(u)

    for it in range(max_iter):
        delta = 0.0

        # Interior of u is rows 1..H-2, cols 1..W-2
        # interior_mask is (H-2, W-2): mask[ii, jj] => grid point (ii+1, jj+1)
        for i in range(1, H - 1):
            for j in range(1, W - 1):           # ← inner loop on j → cache-friendly
                if interior_mask[i - 1, j - 1]:
                    new_val = 0.25 * (u[i, j - 1] + u[i, j + 1]
                                    + u[i - 1, j] + u[i + 1, j])
                    diff = new_val - u[i, j]
                    if diff < 0.0:
                        diff = -diff
                    if diff > delta:
                        delta = diff
                    u_new[i, j] = new_val
                else:
                    # Wall or exterior: keep value unchanged
                    u_new[i, j] = u[i, j]

        # Swap: u_new becomes the new u for next iteration
        u, u_new = u_new, u

        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


if __name__ == '__main__':
    N = int(sys.argv[1])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    # JIT warm-up: compile on a tiny dummy problem so we don't count
    # compilation time in the benchmark
    _u_dummy = np.zeros((10, 10))
    _mask_dummy = np.ones((8, 8), dtype=np.bool_)
    jacobi_jit(_u_dummy, _mask_dummy, 5, 1e-6)

    # Now the real run
    t0 = time.perf_counter()
    results = []
    for bid in building_ids:
        u0, interior_mask = load_data(bid)
        u = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, interior_mask)
        results.append((bid, stats))
    t1 = time.perf_counter()

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

    print(f"# elapsed: {t1 - t0:.3f} s, N: {N}, jit_serial",
          file=sys.stderr)