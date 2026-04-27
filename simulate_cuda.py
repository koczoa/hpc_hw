from os.path import join
import sys
import time
import math

import numpy as np
from numba import cuda, float32


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000  # No early stopping per task 8 — always run full count


# ---------- GPU kernel ----------

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    """One Jacobi iteration. Each thread updates one grid cell."""
    i, j = cuda.grid(2)
    H, W = u.shape

    # Skip halo (rows 0, H-1, cols 0, W-1) and any out-of-bounds threads
    if i < 1 or i >= H - 1 or j < 1 or j >= W - 1:
        return

    # interior_mask is shape (H-2, W-2); cell (i,j) ↔ mask (i-1, j-1)
    if interior_mask[i - 1, j - 1]:
        u_new[i, j] = 0.25 * (u[i, j - 1] + u[i, j + 1]
                            + u[i - 1, j] + u[i + 1, j])
    else:
        # Wall or exterior: copy unchanged so ping-pong stays consistent
        u_new[i, j] = u[i, j]


# ---------- Host (CPU) helper ----------

def jacobi_gpu(u_host, mask_host, max_iter):
    """Run max_iter Jacobi iterations on the GPU. Returns final u on host."""
    # Cast to float32 to match GPU strengths (faster, less memory traffic)
    u_host = u_host.astype(np.float32)

    # 1. Move data CPU → GPU (one-time cost, before the loop)
    d_u     = cuda.to_device(u_host)
    d_u_new = cuda.to_device(u_host)        # init same as u
    d_mask  = cuda.to_device(mask_host)     # bool array

    # 2. Configure grid: 16x16 blocks, enough blocks to cover the grid
    TPB = (16, 16)
    BPG = (math.ceil(u_host.shape[0] / TPB[0]),
           math.ceil(u_host.shape[1] / TPB[1]))

    # 3. Launch the kernel max_iter times, ping-pong arrays each iteration
    for _ in range(max_iter):
        jacobi_kernel[BPG, TPB](d_u, d_u_new, d_mask)
        d_u, d_u_new = d_u_new, d_u

    # 4. Sync, then copy result back GPU → CPU
    cuda.synchronize()
    return d_u.copy_to_host()


# ---------- Boilerplate (load, stats, main) ----------

def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
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

    # ---- Warm-up: trigger CUDA kernel compilation on a dummy problem ----
    # Without this, the first floorplan's timing includes ~1s of compile cost.
    _u_dummy = np.zeros((20, 20), dtype=np.float32)
    _mask_dummy = np.ones((18, 18), dtype=np.bool_)
    _ = jacobi_gpu(_u_dummy, _mask_dummy, 5)

    # ---- Real run ----
    t0 = time.perf_counter()
    results = []
    for bid in building_ids:
        u0, interior_mask = load_data(bid)
        u = jacobi_gpu(u0, interior_mask, MAX_ITER)
        stats = summary_stats(u, interior_mask)
        results.append((bid, stats))
    t1 = time.perf_counter()

    # CSV output
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

    print(f"# elapsed: {t1 - t0:.3f} s, N: {N}, cuda_custom",
          file=sys.stderr)