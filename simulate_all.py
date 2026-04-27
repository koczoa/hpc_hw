from os.path import join
import sys
import time
from multiprocessing import Pool

import numpy as np


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:]
                      + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp':    u_interior.mean(),
        'std_temp':     u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


def process_one(bid):
    u0, mask = load_data(bid)
    u = jacobi(u0, mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, mask)
    return bid, stats


if __name__ == '__main__':
    n_workers = int(sys.argv[1])
    out_path = sys.argv[2]

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    print(f"Processing {len(building_ids)} floorplans with {n_workers} workers",
          file=sys.stderr)

    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        results = pool.map(process_one, building_ids, chunksize=1)
    t1 = time.perf_counter()

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    with open(out_path, 'w') as f:
        f.write('building_id,' + ','.join(stat_keys) + '\n')
        for bid, stats in results:
            f.write(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys) + '\n')

    print(f"# elapsed: {t1 - t0:.1f} s, N: {len(building_ids)}, "
          f"workers: {n_workers}", file=sys.stderr)
    print(f"# wrote {out_path}", file=sys.stderr)