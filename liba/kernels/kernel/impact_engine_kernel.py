

import numpy as np
from numba import njit

try:
    from ._cfg import BOUNDSCHECK
except (ImportError, ValueError):
    from _cfg import BOUNDSCHECK

@njit(nogil=True, boundscheck=BOUNDSCHECK)
def _get_multiscale_samples(forces, min_force_threshold=1.0, *args):
    n = len(forces)
    num_extra = len(args)
    samples_list = []

    stack = [(0, n)]
    while len(stack) > 0:
        s, e = stack.pop()
        curr_n = e - s
        if curr_n < 1:
            continue

        curr_f_sum = 0.0
        abs_f_in_interval = 0.0
        extra_sums = np.zeros(num_extra, dtype=np.float64)

        for i in range(s, e):
            f_val = forces[i]
            curr_f_sum += f_val
            abs_f_in_interval += abs(f_val)
            for k in range(num_extra):
                extra_sums[k] += args[k][i]

        sample = np.zeros(1 + num_extra, dtype=np.float64)
        sample[0] = curr_f_sum
        for k in range(num_extra):
            sample[k+1] = extra_sums[k]
        samples_list.append(sample)

        if abs_f_in_interval > min_force_threshold and curr_n >= 2:
            mid = s + curr_n // 2
            stack.append((mid, e))
            stack.append((s, mid))


    num_samples = len(samples_list)
    res = np.zeros((num_samples, 1 + num_extra), dtype=np.float64)
    for i in range(num_samples):
        res[i] = samples_list[i]

    return res


@njit(nogil=True, boundscheck=BOUNDSCHECK)
def _fit_constrained_power_law(forces, price_moves, min_beta, max_beta):


    mask_diff = (forces > 1e-20) & (price_moves > 1e-20)
    if np.any(mask_diff):
        local_max_vol = np.max(forces[mask_diff])
        local_min_move = np.min(price_moves[mask_diff])
    else:
        return 0.0, (min_beta + max_beta) / 2.0, 0.0



    samples =_get_multiscale_samples(forces, 0.1, price_moves)
    s_forces = samples[:, 0]
    s_moves = samples[:, 1]

    mask_s = (s_forces > 1e-20) & (s_moves > 1e-20)
    valid_forces = s_forces[mask_s]
    valid_moves = s_moves[mask_s]


    if len(valid_forces) < 10:
        return 0.0, (min_beta + max_beta) / 2.0, 0.0


    log_x = np.log(valid_forces)
    log_y = np.log(valid_moves)

    n = len(log_x)
    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xx = np.sum(log_x**2)
    sum_xy = np.sum(log_x * log_y)

    denominator = n * sum_xx - sum_x**2


    if abs(denominator) < 1e-12:
        beta = (min_beta + max_beta) / 2.0
    else:
        raw_beta = (n * sum_xy - sum_x * sum_y) / denominator
        beta = max(min_beta, min(raw_beta, max_beta))



    log_alpha = (sum_y - beta * sum_x) / n



    log_alpha_min_bound = np.log(local_min_move) - beta * np.log(local_max_vol)
    log_alpha = max(log_alpha, log_alpha_min_bound)

    alpha = np.exp(log_alpha)


    preds = alpha * (valid_forces ** beta)
    rss = np.sum((valid_moves - preds)**2)

    return alpha, beta, rss

@njit(nogil=True, boundscheck=BOUNDSCHECK)
def _get_multiscale_samples_weighted(forces, min_force_threshold=1.0, *args):
    n = len(forces)
    num_extra = len(args)


    samples_list = []
    weights_list = []


    stack = [(0, n)]

    while len(stack) > 0:
        s, e = stack.pop()
        curr_n = e - s
        if curr_n < 1:
            continue

        curr_f_sum = 0.0
        abs_f_in_interval = 0.0
        extra_sums = np.zeros(num_extra, dtype=np.float64)


        for i in range(s, e):
            f_val = forces[i]
            curr_f_sum += f_val
            abs_f_in_interval += abs(f_val)
            for k in range(num_extra):
                extra_sums[k] += args[k][i]


        sample = np.zeros(1 + num_extra, dtype=np.float64)
        sample[0] = curr_f_sum
        for k in range(num_extra):
            sample[k+1] = extra_sums[k]

        samples_list.append(sample)

        weights_list.append(float(curr_n))


        if abs_f_in_interval > min_force_threshold and curr_n >= 2:
            mid = s + curr_n // 2
            stack.append((mid, e))
            stack.append((s, mid))


    num_samples = len(samples_list)
    res_samples = np.zeros((num_samples, 1 + num_extra), dtype=np.float64)
    res_weights = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        res_samples[i] = samples_list[i]
        res_weights[i] = weights_list[i]

    return res_samples, res_weights


@njit(nogil=True, boundscheck=BOUNDSCHECK)
def _fit_constrained_power_law_weighted(forces, price_moves, min_beta, max_beta):


    mask_diff = (forces > 1e-20) & (price_moves > 1e-20)
    if np.any(mask_diff):
        local_max_vol = np.max(forces[mask_diff])
        local_min_move = np.min(price_moves[mask_diff])
    else:
        return 0.0, (min_beta + max_beta) / 2.0, 0.0




    samples, weight =_get_multiscale_samples_weighted(forces, 0.1, price_moves)
    s_forces = samples[:, 0]
    s_moves = samples[:, 1]

    mask_s = (s_forces > 1e-20) & (s_moves > 1e-20)
    valid_forces = s_forces[mask_s]
    valid_moves = s_moves[mask_s]
    w = weight[mask_s]


    if len(valid_forces) < 10:
        return 0.0, (min_beta + max_beta) / 2.0, 0.0


    log_x = np.log(valid_forces)
    log_y = np.log(valid_moves)


    sum_w = np.sum(w)
    sum_wx = np.sum(w * log_x)
    sum_wy = np.sum(w * log_y)
    sum_wxx = np.sum(w * (log_x**2))
    sum_wxy = np.sum(w * (log_x * log_y))


    denominator = sum_w * sum_wxx - sum_wx**2


    if abs(denominator) < 1e-12:
        beta = (min_beta + max_beta) / 2.0
    else:

        raw_beta = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
        beta = max(min_beta, min(raw_beta, max_beta))



    log_alpha = (sum_wy - beta * sum_wx) / sum_w


    log_alpha_min_bound = np.log(local_min_move) - beta * np.log(local_max_vol)
    log_alpha = max(log_alpha, log_alpha_min_bound)
    alpha = np.exp(log_alpha)


    preds = alpha * (valid_forces ** beta)

    rss = np.sum(w * (valid_moves - preds)**2)

    return alpha, beta, rss

@njit(nogil=True, boundscheck=BOUNDSCHECK)
def inverse(
    time_square: np.ndarray,
    price_moves: np.ndarray,
    channel_indices: np.ndarray,
    beta_range: tuple = (0.35, 0.8)
) -> np.ndarray:
    price_moves = np.abs(price_moves).astype(np.float64)
    k = channel_indices.shape[0]
    results = np.zeros((k, 3), dtype=np.float64)
    min_beta, max_beta = beta_range


    for i in range(k):
        col_idx = channel_indices[i]
        forces = np.abs(time_square[:, col_idx])

        alpha, beta, _ = _fit_constrained_power_law_weighted(forces, price_moves, min_beta, max_beta)

        results[i, 0] = alpha
        results[i, 1] = beta

    return results

@njit(nogil=True, boundscheck=BOUNDSCHECK)
def inverse2(
    time_square: np.ndarray,
    price_moves: np.ndarray,
    channel_indices: np.ndarray,
    beta_range: tuple = (0.35, 0.8),
    consider_offset: bool = True,
    steps: int = 10
) -> np.ndarray:
    price_moves_abs = np.abs(price_moves).astype(np.float64)
    k = channel_indices.shape[0]

    results = np.zeros((k, 3), dtype=np.float64)
    min_beta, max_beta = beta_range

    for i in range(k):
        col_idx = channel_indices[i]
        forces = time_square[:, col_idx].astype(np.float64)


        mask = (forces > 1e-20) & (price_moves_abs > 1e-20)
        v_active = forces[mask]
        p_active = price_moves_abs[mask]

        if len(v_active) < 15:
            results[i, 0] = 0.0
            results[i, 1] = (min_beta + max_beta) / 2.0
            results[i, 2] = 0.0
            continue


        v_min = np.min(v_active)
        best_rss = 1e18
        best_alpha, best_beta, best_vt = 0.0, 0.0, 0.0

        steps = steps if consider_offset else 1

        for s in range(steps):

            vt = v_min*0.999 * (s / steps) if consider_offset else 0.0



            a, b, curr_rss = _fit_constrained_power_law(v_active - vt, p_active, min_beta, max_beta)


            if curr_rss < best_rss:
                best_rss = curr_rss
                best_alpha, best_beta, best_vt = a, b, vt

        results[i, 0] = best_alpha
        results[i, 1] = best_beta
        results[i, 2] = best_vt

    return results

@njit(nogil=True, boundscheck=BOUNDSCHECK)
def forward(
    active_net_flow: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    n_time, n_cols = active_net_flow.shape
    impact = np.empty((n_time, n_cols), dtype=np.float64)


    for j in range(n_cols):

        alpha = params[j, 0]
        beta  = params[j, 1]
        vt    = params[j, 2]


        if alpha == 0:
            impact[:, j] = 0.0
            continue

        for i in range(n_time):
            flow = active_net_flow[i, j]
            abs_flow = abs(flow)


            if abs_flow <= vt:
                impact[i, j] = 0.0
            else:


                impact[i, j] = np.sign(flow) * alpha * ((abs_flow - vt) ** beta)

    return impact
