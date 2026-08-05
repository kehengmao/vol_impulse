

import numpy as np
from numba import njit

from enum import IntEnum, unique

try:
    from ._cfg import BOUNDSCHECK
except (ImportError, ValueError):
    from _cfg import BOUNDSCHECK

P_ANCHOR = int(0)

@unique
class PulsaDataChannel(IntEnum):


    limit_order_3d = 0
    hidden_trade_3d = 1
    price_mask_3d = 2
    active_trade_3d = 3
    iceberg_est_3d = 4
    total_limit_est_3d = 5
    mean_limit_depth_3d = 6
    mean_est_depth_3d = 7




    vol_delta = 8
    bid1_idx = 9
    ask1_idx = 10
    active_buy = 11
    active_sell = 12
    active_net_vol = 13


    avg_bid_limit = 14
    avg_ask_limit = 15
    avg_bid_est = 16
    avg_ask_est = 17


    delta_bid_limit = 18
    delta_ask_limit = 19
    delta_bid_est = 20
    delta_ask_est = 21


    net_limit_level = 22
    depth_limit_level = 23
    net_est_level = 24
    depth_est_level = 25
    net_limit_mom = 26
    net_est_mom = 27


    active_intensity = 28
    explicit_mom_int = 29
    hidden_mom_int = 30
    flow_press_explicit = 31
    flow_press_hidden = 32

    trade_price_idx = 33
    trade_price_diff = 34


@unique
class PulsaInputChannel(IntEnum):

    input_bid_price = 0
    input_ask_price = 1
    input_trade_price = 2
    input_bid_vol = 3
    input_ask_vol = 4
    input_total_vol = 5

PULSA_CHANNEL_COUNT = len(PulsaDataChannel)

@njit(inline='always')
def _compute_trade_direction(
    curr_lp: int,
    prev_lp: int,
    prev_bx: int,
    prev_ax: int
) -> int:


    if curr_lp >= prev_ax:
        return 1
    if curr_lp <= prev_bx:
        return -1




    return 0

@njit(inline='always')
def _calculate_iceberg_ratio(
    v_hidden_abs: float,
    v_explicit: float,
    iceberg_limit_factor: float
) -> float:
    if v_explicit > 0:
        ratio = v_hidden_abs / v_explicit
    else:
        ratio = iceberg_limit_factor


    return min(ratio, iceberg_limit_factor) + 1.0


@njit(nogil=True, boundscheck=BOUNDSCHECK)
def render_liquidity_cube_integrated(data, iceberg_limit_factor=2.0):

    T_input = data.shape[0]
    if T_input < 2:
        raise ValueError("Input data must have at least 2 rows for differential rendering.")


    bid_price_col   = data[:, PulsaInputChannel.input_bid_price.value]
    ask_price_col   = data[:, PulsaInputChannel.input_ask_price.value]
    trade_price_col = data[:, PulsaInputChannel.input_trade_price.value]
    bid_vol_col     = data[:, PulsaInputChannel.input_bid_vol.value]
    ask_vol_col     = data[:, PulsaInputChannel.input_ask_vol.value]
    total_vol_col   = data[:, PulsaInputChannel.input_total_vol.value]

    max_b = np.max(bid_price_col)
    max_a = np.max(ask_price_col)
    max_lp = np.max(trade_price_col)


    price_range = int(max(max_b, max(max_a, max_lp)))


    if price_range < 1:
        price_range = 1

    P_max = price_range + 1
    cube = np.zeros((T_input, P_max, PULSA_CHANNEL_COUNT), dtype=np.float64)

    last_avg_b0, last_avg_a0 = 0.0, 0.0
    last_avg_b5, last_avg_a5 = 0.0, 0.0

    for i in range(1, T_input):

        bx, ax = np.int64(bid_price_col[i]), np.int64(ask_price_col[i])
        prev_bx, prev_ax = np.int64(bid_price_col[i-1]), np.int64(ask_price_col[i-1])
        curr_lp, prev_lp = np.int64(trade_price_col[i]), np.int64(trade_price_col[i-1])
        delta_v_raw = float(total_vol_col[i] - total_vol_col[i-1])




        v_b = float(bid_vol_col[i])
        cube[i, bx, PulsaDataChannel.limit_order_3d.value] = v_b
        cube[i, bx, PulsaDataChannel.total_limit_est_3d.value]   = v_b
        cube[i, bx, PulsaDataChannel.price_mask_3d.value]  = 1
        cube[i, P_ANCHOR, PulsaDataChannel.bid1_idx.value] = bx

        v_a = -float(ask_vol_col[i])
        cube[i, ax, PulsaDataChannel.limit_order_3d.value] = v_a
        cube[i, ax, PulsaDataChannel.total_limit_est_3d.value]   = v_a
        cube[i, ax, PulsaDataChannel.price_mask_3d.value]  = 2
        cube[i, P_ANCHOR, PulsaDataChannel.ask1_idx.value] = ax


        cube[i, curr_lp, PulsaDataChannel.price_mask_3d.value] = 3
        cube[i, P_ANCHOR, PulsaDataChannel.vol_delta.value] = delta_v_raw


        if delta_v_raw > 0:
            v_dir_int = _compute_trade_direction(curr_lp, prev_lp, prev_bx, prev_ax)
            v_dir = float(v_dir_int)

            if v_dir > 0: cube[i, P_ANCHOR, PulsaDataChannel.active_buy.value] = delta_v_raw
            elif v_dir < 0: cube[i, P_ANCHOR, PulsaDataChannel.active_sell.value] = delta_v_raw


            v_net = delta_v_raw * v_dir
            cube[i, P_ANCHOR, PulsaDataChannel.active_net_vol.value] = v_net
            cube[i, curr_lp, PulsaDataChannel.active_trade_3d.value] = v_net



            if v_dir_int == 1 and curr_lp >= prev_ax:
                explicit_v = min(delta_v_raw, float(ask_vol_col[i-1]))
                rem_v = delta_v_raw - explicit_v


                cube[i, prev_ax, PulsaDataChannel.limit_order_3d.value] -= explicit_v
                cube[i, prev_ax, PulsaDataChannel.total_limit_est_3d.value] -= explicit_v


                if rem_v > 0:
                    if curr_lp == prev_ax:
                        cube[i, prev_ax, PulsaDataChannel.limit_order_3d.value] -= rem_v
                        cube[i, prev_ax, PulsaDataChannel.total_limit_est_3d.value] -= rem_v
                    else:
                        start_p = prev_ax + 1
                        dist = curr_lp - start_p
                        num_ticks = dist + 1
                        fill = rem_v / num_ticks
                        for s in range(num_ticks):
                            target_idx = start_p + s
                            if 0 <= target_idx < P_max:
                                cube[i, target_idx, PulsaDataChannel.limit_order_3d.value] -= fill
                                cube[i, target_idx, PulsaDataChannel.total_limit_est_3d.value] -= fill

            elif v_dir_int == -1 and curr_lp <= prev_bx:
                explicit_v = min(delta_v_raw, float(bid_vol_col[i-1]))
                rem_v = delta_v_raw - explicit_v


                cube[i, prev_bx, PulsaDataChannel.limit_order_3d.value] += explicit_v
                cube[i, prev_bx, PulsaDataChannel.total_limit_est_3d.value] += explicit_v


                if rem_v > 0:
                    if curr_lp == prev_bx:
                        cube[i, prev_bx, PulsaDataChannel.limit_order_3d.value] += rem_v
                        cube[i, prev_bx, PulsaDataChannel.total_limit_est_3d.value] += rem_v
                    else:
                        start_p = prev_bx - 1
                        dist = start_p - curr_lp
                        num_ticks = dist + 1
                        fill = rem_v / num_ticks
                        for s in range(num_ticks):
                            target_idx = start_p - s
                            if 0 <= target_idx < P_max:
                                cube[i, target_idx, PulsaDataChannel.limit_order_3d.value] += fill
                                cube[i, target_idx, PulsaDataChannel.total_limit_est_3d.value] += fill





            if v_dir != 0 and (curr_lp == prev_ax or curr_lp == prev_bx):
                v_explicit = float(ask_vol_col[i-1]) if v_dir > 0 else float(bid_vol_col[i-1])
                v_hidden_pure = max(0.0, delta_v_raw - v_explicit)
                if v_hidden_pure > 0:
                    cube[i, curr_lp, PulsaDataChannel.hidden_trade_3d.value] = -v_dir * v_hidden_pure
                    ratio = _calculate_iceberg_ratio(v_hidden_pure, v_explicit, iceberg_limit_factor)
                    est_depth_val = delta_v_raw * ratio * (-v_dir)
                    cube[i, curr_lp, PulsaDataChannel.iceberg_est_3d.value] = est_depth_val

                    cube[i, curr_lp, PulsaDataChannel.total_limit_est_3d.value] = est_depth_val


        sum_b0, count_b0 = 0.0, 0
        sum_a0, count_a0 = 0.0, 0
        sum_b5, count_b5 = 0.0, 0
        sum_a5, count_a5 = 0.0, 0

        for p in range(P_max):
            c0_val = cube[i, p, PulsaDataChannel.limit_order_3d.value]
            if c0_val > 0:
                sum_b0 += c0_val; count_b0 += 1
            elif c0_val < 0:
                sum_a0 += abs(c0_val); count_a0 += 1

            c5_val = cube[i, p, PulsaDataChannel.total_limit_est_3d.value]
            if c5_val > 0:
                sum_b5 += c5_val; count_b5 += 1
            elif c5_val < 0:
                sum_a5 += abs(c5_val); count_a5 += 1

        avg_b0 = sum_b0 / count_b0 if count_b0 > 0 else 0.0
        avg_a0 = sum_a0 / count_a0 if count_a0 > 0 else 0.0
        avg_b5 = sum_b5 / count_b5 if count_b5 > 0 else 0.0
        avg_a5 = sum_a5 / count_a5 if count_a5 > 0 else 0.0




        cube[i, bx, PulsaDataChannel.mean_limit_depth_3d.value] = avg_b0
        cube[i, bx, PulsaDataChannel.mean_est_depth_3d.value] = avg_b5
        cube[i, P_ANCHOR, PulsaDataChannel.avg_bid_limit.value] = avg_b0
        cube[i, P_ANCHOR, PulsaDataChannel.avg_bid_est.value] = avg_b5

        cube[i, ax, PulsaDataChannel.mean_limit_depth_3d.value] = -avg_a0
        cube[i, ax, PulsaDataChannel.mean_est_depth_3d.value] = -avg_a5
        cube[i, P_ANCHOR, PulsaDataChannel.avg_ask_limit.value] = avg_a0
        cube[i, P_ANCHOR, PulsaDataChannel.avg_ask_est.value] = avg_a5


        cube[i, P_ANCHOR, PulsaDataChannel.delta_bid_limit.value] = avg_b0 - last_avg_b0
        cube[i, P_ANCHOR, PulsaDataChannel.delta_ask_limit.value] = avg_a0 - last_avg_a0
        cube[i, P_ANCHOR, PulsaDataChannel.delta_bid_est.value] = avg_b5 - last_avg_b5
        cube[i, P_ANCHOR, PulsaDataChannel.delta_ask_est.value] = avg_a5 - last_avg_a5

        cube[i, P_ANCHOR, PulsaDataChannel.net_limit_level.value] = avg_b0 - avg_a0
        cube[i, P_ANCHOR, PulsaDataChannel.depth_limit_level.value] = avg_b0 + avg_a0
        cube[i, P_ANCHOR, PulsaDataChannel.net_est_level.value] = avg_b5 - avg_a5
        cube[i, P_ANCHOR, PulsaDataChannel.depth_est_level.value] = avg_b5 + avg_a5

        cube[i, P_ANCHOR, PulsaDataChannel.trade_price_idx.value] = curr_lp
        cube[i, P_ANCHOR, PulsaDataChannel.trade_price_diff.value] = curr_lp - prev_lp

        denom = avg_b0 + avg_a0
        denom_est = avg_b5 + avg_a5
        v_net_val = cube[i, P_ANCHOR, PulsaDataChannel.active_net_vol.value]

        if denom > 1e-8:
            cube[i, P_ANCHOR, PulsaDataChannel.active_intensity.value] = v_net_val / denom

            exp_mom = (avg_b0 - last_avg_b0) - (avg_a0 - last_avg_a0)
            cube[i, P_ANCHOR, PulsaDataChannel.explicit_mom_int.value] = exp_mom / denom
            cube[i, P_ANCHOR, PulsaDataChannel.flow_press_explicit.value] = (v_net_val + exp_mom) / denom

        if denom_est > 1e-8:
            hid_mom = (avg_b5 - last_avg_b5) - (avg_a5 - last_avg_a5)
            cube[i, P_ANCHOR, PulsaDataChannel.hidden_mom_int.value] = hid_mom / denom_est
            cube[i, P_ANCHOR, PulsaDataChannel.flow_press_hidden.value] = (v_net_val + hid_mom) / denom_est

        last_avg_b0, last_avg_a0 = avg_b0, avg_a0
        last_avg_b5, last_avg_a5 = avg_b5, avg_a5


    return np.ascontiguousarray(cube[1:])
