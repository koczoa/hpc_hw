from os.path import join
import sys
import numpy as np
import numba as nb
from numba import cuda
import time

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    rows, cols = interior_mask.shape

    if i < rows and j < cols:
        if interior_mask[i, j]:
            # +1 offset for border padding in u
            u_new[i+1, j+1] = 0.25 * (
                u[i+1, j]   +  # left
                u[i+1, j+2] +  # right
                u[i,   j+1] +  # up
                u[i+2, j+1]    # down
            )
        else:
            # Non-interior points stay fixed
            u_new[i+1, j+1] = u[i+1, j+1]

@cuda.jit
def copy_border(u, u_new):
    # Copy the 1-cell border padding from u to u_new unchanged
    i = cuda.grid(1)
    size = u.shape[0]
    if i < size:
        u_new[i, 0]      = u[i, 0]
        u_new[i, -1]     = u[i, -1]
        u_new[0, i]      = u[0, i]
        u_new[-1, i]     = u[-1, i]

def jacobi_cuda(u, interior_mask, max_iter):
    # Transfer to GPU
    u_d      = cuda.to_device(u.astype(np.float64))
    u_new_d  = cuda.to_device(np.copy(u).astype(np.float64))
    mask_d   = cuda.to_device(interior_mask)

    # Grid/block dimensions — 16x16 threads per block
    threads_per_block = (16, 16)
    blocks_x = (interior_mask.shape[0] + 15) // 16
    blocks_y = (interior_mask.shape[1] + 15) // 16
    blocks_per_grid = (blocks_x, blocks_y)

    border_threads = 512 + 2
    border_blocks  = (border_threads + 31) // 32

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_d, u_new_d, mask_d)
        copy_border[border_blocks, 32](u_d, u_new_d)
        cuda.synchronize()
        # Swap buffers
        u_d, u_new_d = u_new_d, u_d

    return u_d.copy_to_host()

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

    # Warm up CUDA kernel
    print("Warming up CUDA kernel...", flush=True)
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cuda(u0, mask0, max_iter=1)
    print("CUDA ready.", flush=True)

    t0 = time.time()
    all_results = []
    for bid in building_ids:
        u, interior_mask = load_data(LOAD_DIR, bid)
        u = jacobi_cuda(u, interior_mask, MAX_ITER)
        stats = summary_stats(u, interior_mask)
        all_results.append((bid, stats))
    elapsed = time.time() - t0

    print(f"# N={N}, time={elapsed:.2f}s", flush=True)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in all_results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))