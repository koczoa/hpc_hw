from os.path import join
import numpy as np
import cupy as cp
import nvtx
import time

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_cupy(u, interior_mask, max_iter, atol=1e-4):
    u_d    = cp.asarray(u, dtype=cp.float64)
    mask_d = cp.asarray(interior_mask)

    for i in range(max_iter):
        with nvtx.annotate(f"iter", color="blue"):
            u_new = 0.25 * (u_d[1:-1, :-2] + u_d[1:-1, 2:]
                          + u_d[:-2, 1:-1] + u_d[2:, 1:-1])
            u_new_interior = u_new[mask_d]
            delta = cp.abs(u_d[1:-1, 1:-1][mask_d] - u_new_interior).max()
            u_d[1:-1, 1:-1][mask_d] = u_new_interior
            if float(delta) < atol:   # <-- CPU/GPU sync point!
                break

    return cp.asnumpy(u_d)

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    bid = open(join(LOAD_DIR, 'building_ids.txt')).readline().strip()

    u, interior_mask = load_data(LOAD_DIR, bid)

    # Warm up
    _ = jacobi_cupy(u, interior_mask, max_iter=1)

    # Profile just 100 iterations
    jacobi_cupy(u, interior_mask, max_iter=100, atol=0.0)