from os.path import join
import sys
import numpy as np
import numba as nb
import time

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@nb.jit(nopython=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    rows, cols = interior_mask.shape
    for iteration in range(max_iter):
        delta = 0.0
        for i in range(rows):
            for j in range(cols):
                if interior_mask[i, j]:
                    u_new = 0.25 * (u[i, j-1+1] + u[i, j+1+1]
                                  + u[i-1+1, j+1] + u[i+1+1, j+1])
                    # +1 offset because u has a 1-cell border padding
                    diff = abs(u[i+1, j+1] - u_new)
                    if diff > delta:
                        delta = diff
                    u[i+1, j+1] = u_new
        if delta < atol:
            break
    return u

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

    # Warm up JIT on first building (compilation not counted in timing)
    u0, interior_mask0 = load_data(LOAD_DIR, building_ids[0])
    print("Warming up JIT...", flush=True)
    _ = jacobi_numba(np.copy(u0), interior_mask0, max_iter=1)
    print("JIT ready.", flush=True)

    t0 = time.time()
    all_results = []
    for bid in building_ids:
        u, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi_numba(u, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, interior_mask)
        all_results.append((bid, stats))
    elapsed = time.time() - t0

    print(f"# N={N}, time={elapsed:.2f}s", flush=True)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in all_results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))