from os.path import join
import sys
import time

import numpy as np
import cupy as cp


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_cupy(u, interior_mask, max_iter, atol):
    """Same algorithm as reference, every op runs on the GPU.
    Inputs/outputs are CuPy arrays."""
    u = cp.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:]
                      + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = cp.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u


def summary_stats(u_host, interior_mask_host):
    """Stats on CPU using NumPy."""
    u_interior = u_host[1:-1, 1:-1][interior_mask_host]
    return {
        'mean_temp':    u_interior.mean(),
        'std_temp':     u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


if __name__ == '__main__':
    N = int(sys.argv[1])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    # Warm-up: trigger CuPy/CUDA initialization and kernel compilation
    _u_dummy = cp.zeros((32, 32), dtype=cp.float32)
    _mask_dummy = cp.ones((30, 30), dtype=bool)
    _ = jacobi_cupy(_u_dummy, _mask_dummy, 5, 1e-6)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    results = []
    for bid in building_ids:
        # Load on CPU
        u0_host, mask_host = load_data(bid)

        # Move to GPU (use float32 for speed)
        u0_gpu   = cp.asarray(u0_host, dtype=cp.float32)
        mask_gpu = cp.asarray(mask_host)

        # Run simulation on GPU
        u_gpu = jacobi_cupy(u0_gpu, mask_gpu, MAX_ITER, ABS_TOL)

        # Move result back to CPU for stats
        u_host = cp.asnumpy(u_gpu)
        stats = summary_stats(u_host, mask_host)
        results.append((bid, stats))

    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

    print(f"# elapsed: {t1 - t0:.3f} s, N: {N}, cupy",
          file=sys.stderr)