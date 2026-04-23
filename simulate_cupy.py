from os.path import join
import sys
import numpy as np
import cupy as cp
import time

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_cupy(u, interior_mask, max_iter, atol=1e-4):
    # Transfer to GPU
    u_d    = cp.asarray(u, dtype=cp.float64)
    mask_d = cp.asarray(interior_mask)

    for i in range(max_iter):
        u_new = 0.25 * (u_d[1:-1, :-2] + u_d[1:-1, 2:]
                      + u_d[:-2, 1:-1] + u_d[2:, 1:-1])
        u_new_interior = u_new[mask_d]
        delta = cp.abs(u_d[1:-1, 1:-1][mask_d] - u_new_interior).max()
        u_d[1:-1, 1:-1][mask_d] = u_new_interior
        if delta < atol:
            break

    return cp.asnumpy(u_d)

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {'mean_temp': mean_temp, 'std_temp': std_temp,
            'pct_above_18': pct_above_18, 'pct_below_15': pct_below_15}

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Warm up CuPy
    print("Warming up CuPy...", flush=True)
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cupy(u0, mask0, max_iter=1)
    print("CuPy ready.", flush=True)

    t0 = time.time()
    all_results = []
    for bid in building_ids:
        u, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi_cupy(u, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, interior_mask)
        all_results.append((bid, stats))
    elapsed = time.time() - t0

    print(f"# N={N}, time={elapsed:.2f}s", flush=True)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in all_results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))