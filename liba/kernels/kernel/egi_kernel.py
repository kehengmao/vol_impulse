
import numpy as np
from numba import njit

from enum import IntEnum, unique, auto

try:
    from .integrals_kernel import n_get_next_head, get_mean_by_integral_logic
    from ._cfg import BOUNDSCHECK
except (ImportError, ValueError):
    from integrals_kernel import n_get_next_head, get_mean_by_integral_logic
    from _cfg import BOUNDSCHECK

@unique
class EgiIntegralChannel(IntEnum):
    real_normal = 0
    real_abs = auto()

    expect_normal = auto()
    expect_abs = auto()

    real_expect_gap_normal = auto()
    real_expect_gap_abs = auto()

    expect_scale_normal = auto()
    expect_scale_abs = auto()

    real_expect_scale_gap_normal = auto()
    real_expect_scale_gap_abs = auto()


@unique
class EgiDataChannel(IntEnum):
    real_value = 0
    real_delta = auto()
    expect_delta = auto()
    expect_delta_scale = auto()
    real_expect_delta_gap = auto()
    real_expect_delta_scale_gap = auto()
    abs_time = auto()
    egi_start_abs_time = auto()
    egi_end_abs_time = auto()
    egi_peak_abs_time = auto()
    egi_value = auto()
    egi_dir = auto()
    egi_dir_intuitive = auto()
    egi_force_dir = auto()
    real_displace = auto()
    estimated_displace = auto()

    raw_extremas = auto()
    full_extremas = auto()
    extremas = auto()

    full_extremas2 = auto()
    extremas2 = auto()



@njit(inline='always', boundscheck=BOUNDSCHECK)
def _calculate_gi(d, t, path_sum):
    if t <= 0 or path_sum <= 1e-9:
        return 0.0


    c = min(1.0, d / path_sum)



    exponent = 0.5 - c / 2.0


    return d / (t ** exponent)

@njit(inline='always', boundscheck=BOUNDSCHECK)
def _gi_to_egi(gi, local_baseline, global_baseline):
    denominator = max(global_baseline, local_baseline)
    if denominator < 1e-9:
        return 0.0
    return max(0.0, (gi - local_baseline) / denominator)



@njit(fastmath=True, inline='always', boundscheck=BOUNDSCHECK)
def _get_gamma(integrals: np.ndarray, start: int, end: int) -> float:
    price_gap = integrals[end, EgiIntegralChannel.real_abs.value] - integrals[start, EgiIntegralChannel.real_abs.value]


    expect_gap = integrals[end, EgiIntegralChannel.expect_abs.value] - integrals[start, EgiIntegralChannel.expect_abs.value]


    if abs(expect_gap) < 1e-12:
        return 1.0


    return price_gap / expect_gap

@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def _update_local_scale(
    data: np.ndarray, # [T, C]
    integrals: np.ndarray, # [T+1 C]
    t_head: int,
    logic_start: int,
    logic_end: int,
):
    capacity = data.shape[0]
    integral_capacity = integrals.shape[0]


    phy_start = n_get_next_head(t_head, logic_start, integral_capacity)
    phy_end = n_get_next_head(t_head, logic_end, integral_capacity)


    gamma = _get_gamma(integrals, phy_start, phy_end)


    length = logic_end - logic_start


    for t in range(length):

        logic_t = logic_start + t

        idx = n_get_next_head(t_head, logic_t, capacity)

        if t_head + logic_t < 0:
            continue

        scaled = data[idx, EgiDataChannel.expect_delta.value] * gamma
        gap = data[idx, EgiDataChannel.real_delta.value] - scaled

        data[idx, EgiDataChannel.real_expect_delta_scale_gap.value] = gap
        data[idx, EgiDataChannel.expect_delta_scale.value] = scaled


        jdx = n_get_next_head(t_head, logic_t + 1, integral_capacity)


        jdx_prev = n_get_next_head(t_head, logic_t, integral_capacity)

        integrals[jdx, EgiIntegralChannel.expect_scale_normal.value] = integrals[jdx_prev, EgiIntegralChannel.expect_scale_normal.value] + scaled
        integrals[jdx, EgiIntegralChannel.expect_scale_abs.value] = integrals[jdx_prev, EgiIntegralChannel.expect_scale_abs.value] + abs(scaled)

        integrals[jdx, EgiIntegralChannel.real_expect_scale_gap_normal.value] = integrals[jdx_prev, EgiIntegralChannel.real_expect_scale_gap_normal.value] + gap
        integrals[jdx, EgiIntegralChannel.real_expect_scale_gap_abs.value] = integrals[jdx_prev, EgiIntegralChannel.real_expect_scale_gap_abs.value] + abs(gap)


@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def _process_single_peak_initial(
    data: np.ndarray,
    integrals: np.ndarray,
    t_head: int,
    peak_logic: int,
    logic_end: int,
    peak_type: float,
    global_baseline: float,
    min_window: int,
    max_window: int,
    minimal_tick: float
):
    capacity = data.shape[0]
    integral_capacity = integrals.shape[0]


    anchor = peak_logic
    start_offset = anchor - max_window

    gap_idx = EgiIntegralChannel.real_expect_scale_gap_normal.value
    gap_abs_idx = EgiIntegralChannel.real_expect_scale_gap_abs.value
    real_idx = EgiIntegralChannel.real_normal.value
    real_abs_idx = EgiIntegralChannel.real_abs.value
    expect_idx = EgiIntegralChannel.expect_scale_normal.value
    expect_scale_abs_idx = EgiIntegralChannel.expect_scale_abs.value

    egi_start_logic_idx = EgiDataChannel.egi_start_abs_time.value
    egi_end_logic_idx = EgiDataChannel.egi_end_abs_time.value
    eigi_peak_logic_idx = EgiDataChannel.egi_peak_abs_time.value
    egi_value_idx = EgiDataChannel.egi_value.value
    egi_dir_idx = EgiDataChannel.egi_dir.value
    egi_dir_intuitive_idx = EgiDataChannel.egi_dir_intuitive.value
    egi_force_dir_idx = EgiDataChannel.egi_force_dir.value
    real_displace_idx = EgiDataChannel.real_displace.value
    exp_displace_idx = EgiDataChannel.estimated_displace.value

    local_baseline = get_mean_by_integral_logic(integrals[:, gap_abs_idx], start_offset, anchor, t_head)

    anchor_phy = n_get_next_head(t_head, anchor, capacity)

    max_gi = 0.0
    best_js = anchor
    best_je = anchor

    end_offset = min(anchor + max_window, logic_end - 1)

    egi_d = 0
    for je in range(anchor, end_offset + 1):
        je_phy_integ = n_get_next_head(t_head, (je + 1), integral_capacity)

        for js in range(start_offset, anchor + 1):
            t = je - js
            js_phy = n_get_next_head(t_head, js, integral_capacity)

            diff = integrals[je_phy_integ, gap_idx] - integrals[js_phy, gap_idx]

            displace_micro = integrals[je_phy_integ, real_idx] - integrals[js_phy, real_idx]
            max_displace = integrals[je_phy_integ, real_abs_idx] - integrals[js_phy, real_abs_idx]
            max_displace_expected = integrals[je_phy_integ, expect_scale_abs_idx] - integrals[js_phy, expect_scale_abs_idx]

            displace_micro_expected = integrals[je_phy_integ, expect_idx] - integrals[js_phy, expect_idx]

            main_f, sub_f = (displace_micro, displace_micro_expected) if abs(displace_micro) > abs(displace_micro_expected) else (displace_micro_expected, displace_micro)


            path_sum = integrals[je_phy_integ, gap_abs_idx] - integrals[js_phy, gap_abs_idx]


            if path_sum > 1e-9 and min_window <= t <= max_window and max_displace >= minimal_tick and max_displace_expected >= minimal_tick:
                gi = _calculate_gi(abs(diff), t, path_sum)

                if gi > max_gi:
                    max_gi = gi
                    best_js = js
                    best_je = je
                    if (main_f - sub_f) > 0:
                        egi_d = 1
                    elif (main_f - sub_f) < 0:
                        egi_d = -1
                    else:
                        egi_d = 0

    if (best_js == anchor and best_je == anchor) or abs(max_gi) < 1e-12:
        return


    abs_best_js_logic = best_js + t_head
    abs_best_je_logic = best_je + t_head
    abs_best_peak_logic = anchor + t_head

    data[anchor_phy, egi_start_logic_idx] = abs_best_js_logic
    data[anchor_phy, egi_end_logic_idx] = abs_best_je_logic
    data[anchor_phy, eigi_peak_logic_idx] = abs_best_peak_logic

    data[anchor_phy, egi_value_idx] = _gi_to_egi(max_gi, local_baseline, global_baseline)

    final_start_phy = n_get_next_head(t_head, best_js, integral_capacity)
    final_end_phy_integ = n_get_next_head(t_head, best_je + 1, integral_capacity)

    real_displace = integrals[final_end_phy_integ, real_idx] - integrals[final_start_phy, real_idx]
    expect_displace = integrals[final_end_phy_integ, expect_idx] - integrals[final_start_phy, expect_idx]


    data[anchor_phy, egi_force_dir_idx] = 1 if abs(expect_displace) > abs(real_displace) else 0
    data[anchor_phy, real_displace_idx] = real_displace
    data[anchor_phy, exp_displace_idx] = expect_displace
    data[anchor_phy, egi_dir_idx] = egi_d
    data[anchor_phy, egi_dir_intuitive_idx] = -egi_d


@njit(nogil=True, fastmath=True, boundscheck=BOUNDSCHECK)
def egi_update(
    data: np.ndarray,
    integrals: np.ndarray, # [T+1 C]
    t_head: int,
    increment: int,
    global_baseline: float,
    min_window: int,
    max_window: int,
    minimal_tick: float
):
    capacity = data.shape[0]

    lookback_window = max_window + 1

    logic_start = -lookback_window
    logic_end = increment

    raw_extremas_idx = EgiDataChannel.raw_extremas.value

    _update_local_scale(data, integrals, t_head, logic_start, logic_end)

    for i in range(logic_start, logic_end):
        i_phy = n_get_next_head(t_head, i, capacity)

        if t_head + i < 0:
            continue

        ext_val_current = data[i_phy, raw_extremas_idx]

        if (ext_val_current > 0.5 or ext_val_current < -0.5):
            _process_single_peak_initial(
                data,
                integrals,
                t_head = t_head,
                peak_logic=i,
                logic_end=logic_end,
                peak_type=ext_val_current,
                global_baseline=global_baseline,
                min_window=min_window,
                max_window=max_window,
                minimal_tick = minimal_tick
            )
