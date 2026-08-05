import numpy as np
from numba import njit

try:
    from ._cfg import BOUNDSCHECK
except (ImportError, ValueError):
    from _cfg import BOUNDSCHECK

@njit(inline='always')
def n_get_next_head(current_head: int, offset: int, capacity: int) -> int:
    return (current_head + offset + capacity) % capacity

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def update_integral(out_integral: np.ndarray, current: float, increments: np.ndarray, head: int, is_abs: bool):
    n = len(increments)

    for i in range(n):


        current += abs(increments[i]) if is_abs else increments[i]


        j = n_get_next_head(head, i+1, out_integral.shape[0])

        out_integral[j] = current

    return current


@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def update_integral_multi(out_integrals: np.ndarray, currents: np.ndarray, increments_batch: np.ndarray, heads: np.ndarray, channels: np.ndarray, is_abs: bool):
    for i in range(len(channels)):
        c = channels[i]

        currents[c] = update_integral(
            out_integrals[:, c],
            currents[c],
            increments_batch[:, i],
            heads[c],
            is_abs
        )
    return currents

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def update_integral_normal(out_integral: np.ndarray, current: float, increments: np.ndarray, head: int):
    return update_integral(out_integral, current, increments, head, False)

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def update_integral_abs(out_integral: np.ndarray, current: float, increments: np.ndarray, head: int):
    return update_integral(out_integral, current, increments, head, True)

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def shift_integral_channel(out_integrals: np.ndarray, channel_idx: int, shift_amount: float):

    out_integrals[:, channel_idx] -= shift_amount


@njit(fastmath=True, inline='always', boundscheck=BOUNDSCHECK)
def get_mean_by_integral_logic(s_abs, start, end, t_head):

    logical_n = end - start
    if logical_n <= 0:
        return 0.0

    phy_start = n_get_next_head(t_head, start, s_abs.shape[0])
    phy_end = n_get_next_head(t_head, end, s_abs.shape[0])


    return (s_abs[phy_end] - s_abs[phy_start]) / logical_n


@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def fill_original_data(out_integral: np.ndarray, increments: np.ndarray, head: int):
    n = len(increments)
    capacity = out_integral.shape[0]

    for i in range(n):


        j = n_get_next_head(head, i, capacity)
        out_integral[j] = increments[i]

    return increments[n - 1]

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def fill_original_multi(out_integrals: np.ndarray, currents: np.ndarray, increments_batch: np.ndarray, heads: np.ndarray, channels: np.ndarray):
    for i in range(len(channels)):
        c = channels[i]

        currents[c] = fill_original_data(
            out_integrals[:, c],
            increments_batch[:, i],
            heads[c]
        )
    return currents



@njit(nogil=True, boundscheck=BOUNDSCHECK)
def compute_extremums_inplace_circular(new_data: np.ndarray, values: np.ndarray, extremas: np.ndarray, semi_status: np.ndarray, head: int, total_count: int):
    N = len(new_data)
    capacity = values.shape[0]

    for i in range(N):
        curr_idx = n_get_next_head(head, i, capacity)
        prev_idx = n_get_next_head(curr_idx, -1, capacity)

        v_curr = new_data[i]
        values[curr_idx] = v_curr
        extremas[curr_idx] = 0



        if total_count == 0 and i == 0:
            semi_status[curr_idx] = 0
            continue

        v_prev = values[prev_idx]

        if v_curr == v_prev:
            extremas[prev_idx] = 0
            semi_status[curr_idx] = semi_status[prev_idx]
        else:
            left_trend = semi_status[prev_idx]
            if left_trend == 1 and v_prev > v_curr:
                extremas[prev_idx] = 1
            elif left_trend == -1 and v_prev < v_curr:
                extremas[prev_idx] = -1
            else:
                extremas[prev_idx] = 0

            if v_curr > v_prev:
                semi_status[curr_idx] = 1
            else:
                semi_status[curr_idx] = -1



@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def compute_super_extremas_realtime_circular(
    values: np.ndarray,
    extremas: np.ndarray,
    super_extremas: np.ndarray,
    full_extremas: np.ndarray,
    head: int,
    batch_size: int,
    window: int,
    total_count: int
):
    capacity = values.shape[0]
    if batch_size == 0 or window == 0:
        return

    actual_window = min(window, capacity // 2 - 1)
    if actual_window <= 0:
        return

    latest_phy = n_get_next_head(head, batch_size - 1, capacity)

    update_count = batch_size + actual_window
    if update_count > capacity:
        update_count = capacity
    if update_count > total_count:
        update_count = total_count

    for i in range(update_count):
        offset = update_count - 1 - i
        cand_idx = n_get_next_head(latest_phy, -offset, capacity)
        ext = extremas[cand_idx]

        if ext == 0:
            super_extremas[cand_idx] = 0
            full_extremas[cand_idx] = 0
            continue

        val = values[cand_idx]
        is_super = 1

        valid_right_steps = offset if offset < actual_window else actual_window

        left_known = total_count - offset - 1
        max_left_history = capacity - 1 - offset
        actual_left_limit = max_left_history if max_left_history < left_known else left_known
        valid_left_steps = actual_left_limit if actual_left_limit < actual_window else actual_window

        if ext == 1:
            is_left_super = 1
            for step in range(-valid_left_steps, 0):
                check_idx = n_get_next_head(cand_idx, step, capacity)




                if values[check_idx] > val:
                    is_left_super = 0
                    break




            full_extremas[cand_idx] = 1 if is_left_super == 1 else 0

            is_super = is_left_super
            if is_super == 1:
                for step in range(1, valid_right_steps + 1):
                    check_idx = n_get_next_head(cand_idx, step, capacity)


                    if values[check_idx] >= val:
                        is_super = 0
                        break

            super_extremas[cand_idx] = 1 if is_super == 1 else 0

        elif ext == -1:
            is_left_super = 1
            for step in range(-valid_left_steps, 0):
                check_idx = n_get_next_head(cand_idx, step, capacity)

                if values[check_idx] < val:
                    is_left_super = 0
                    break

            full_extremas[cand_idx] = -1 if is_left_super == 1 else 0

            is_super = is_left_super
            if is_super == 1:
                for step in range(1, valid_right_steps + 1):
                    check_idx = n_get_next_head(cand_idx, step, capacity)


                    if values[check_idx] <= val:
                        is_super = 0
                        break

            super_extremas[cand_idx] = -1 if is_super == 1 else 0


@njit(nogil=True, boundscheck=BOUNDSCHECK)
def write_abs_time_circular(abs_time: np.ndarray, head: int, batch_size: int, total_count: int):
    capacity = abs_time.shape[0]
    for i in range(batch_size):
        curr_idx = n_get_next_head(head, i, capacity)
        abs_time[curr_idx] = total_count + i