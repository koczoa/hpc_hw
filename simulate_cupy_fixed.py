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


def jacobi_cupy_fixed(u, interior_mask, max_iter, atol=1e-4, check_every=100):
    u_d    = cp.asarray(u, dtype=cp.float64)
    mask_d = cp.asarray(interior_mask)

    for i in range(max_iter):
        u_new = 0.25 * (u_d[1:-1, :-2] + u_d[1:-1, 2:]
                      + u_d[:-2, 1:-1] + u_d[2:, 1:-1])
        u_new_interior = u_new[mask_d]

        # Only sync with CPU every check_every iterations
        if i % check_every == 0:
            delta = cp.abs(u_d[1:-1, 1:-1][mask_d] - u_new_interior).max()
            if float(delta) < atol:
                break

        u_d[1:-1, 1:-1][mask_d] = u_new_interior

    return cp.asnumpy(u_d)


def sor_cupy_fixed(u, interior_mask, max_iter, atol=1e-4, omega=1.9, check_every=100):
    u_d    = cp.asarray(u, dtype=cp.float64)
    mask_d = cp.asarray(interior_mask)
    H, W = mask_d.shape
    ii = cp.arange(H).reshape(-1, 1)
    jj = cp.arange(W).reshape(1, -1)
    parity_even = ((ii + jj) & 1) == 0
    red_mask   = mask_d &  parity_even
    black_mask = mask_d & ~parity_even
    inner = u_d[1:-1, 1:-1]
    omega_c = 1.0 - omega

    for i in range(max_iter):
        check_now = (i + 1) % check_every == 0
        if check_now:
            u_prev = inner.copy()

        gs = 0.25 * (u_d[1:-1, :-2] + u_d[1:-1, 2:] + u_d[:-2, 1:-1] + u_d[2:, 1:-1])
        inner[:] = cp.where(red_mask, omega_c * inner + omega * gs, inner)

        gs = 0.25 * (u_d[1:-1, :-2] + u_d[1:-1, 2:] + u_d[:-2, 1:-1] + u_d[2:, 1:-1])
        inner[:] = cp.where(black_mask, omega_c * inner + omega * gs, inner)

        if check_now:
            delta = cp.abs(inner - u_prev).max()
            if float(delta) < atol:
                break

    return cp.asnumpy(u_d)


SOLVERS = {
    'jacobi': jacobi_cupy_fixed,
    'sor':    sor_cupy_fixed,
}


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

    # CLI: python simulate_cupy_fixed.py <N> [method] [omega]
    #   method: 'jacobi' (default) or 'sor'
    #   omega:  only used by sor; default 1.9
    N      = int(sys.argv[1])         if len(sys.argv) > 1 else 1
    method = sys.argv[2].lower()      if len(sys.argv) > 2 else 'jacobi'
    omega  = float(sys.argv[3])       if len(sys.argv) > 3 else 1.9

    if method not in SOLVERS:
        print(f"Unknown method '{method}'. Choose from: {list(SOLVERS)}",
              file=sys.stderr)
        sys.exit(1)
    solver = SOLVERS[method]

    # Build solver kwargs (only sor takes omega)
    solver_kwargs = {'omega': omega} if method == 'sor' else {}

    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Warm up
    print(f"Warming up ({method})...", flush=True)
    u0, mask0 = load_data(LOAD_DIR, building_ids[0])
    _ = solver(u0, mask0, max_iter=1, **solver_kwargs)
    print("Ready.", flush=True)

    t0 = time.time()
    all_results = []
    for bid in building_ids:
        u, interior_mask = load_data(LOAD_DIR, bid)
        u = solver(u, interior_mask, MAX_ITER, ABS_TOL, **solver_kwargs)
        stats = summary_stats(u, interior_mask)
        all_results.append((bid, stats))
    elapsed = time.time() - t0

    extra = f", omega={omega}" if method == 'sor' else ""
    print(f"# method={method}{extra}, N={N}, time={elapsed:.2f}s", flush=True)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in all_results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
