"""AOT export specification for circular integral/extrema kernels."""

from numba.pycc import CC

from .integrals_kernel import (
    compute_extremums_inplace_circular,
    compute_super_extremas_realtime_circular,
    fill_original_data,
    fill_original_multi,
    get_mean_by_integral_logic,
    n_get_next_head,
    shift_integral_channel,
    update_integral,
    update_integral_abs,
    update_integral_multi,
    update_integral_normal,
    write_abs_time_circular,
)


cc = CC("integrals_kernel_core")


@cc.export("n_get_next_head", "int64(int64, int64, int64)")
def n_get_next_head_export(current_head, offset, capacity):
    return n_get_next_head(current_head, offset, capacity)


@cc.export("update_integral", "float64(float64[:], float64, float64[:], int64, boolean)")
def update_integral_export(out_integral, current, increments, head, is_abs):
    return update_integral(out_integral, current, increments, head, is_abs)


@cc.export(
    "update_integral_multi",
    "float64[:](float64[:,:], float64[:], float64[:,:], int64[:], int64[:], boolean)",
)
def update_integral_multi_export(
    out_integrals, currents, increments_batch, heads, channels, is_abs
):
    return update_integral_multi(
        out_integrals, currents, increments_batch, heads, channels, is_abs
    )


@cc.export("update_integral_normal", "float64(float64[:], float64, float64[:], int64)")
def update_integral_normal_export(out_integral, current, increments, head):
    return update_integral_normal(out_integral, current, increments, head)


@cc.export("update_integral_abs", "float64(float64[:], float64, float64[:], int64)")
def update_integral_abs_export(out_integral, current, increments, head):
    return update_integral_abs(out_integral, current, increments, head)


@cc.export("shift_integral_channel", "void(float64[:,:], int64, float64)")
def shift_integral_channel_export(out_integrals, channel_idx, shift_amount):
    shift_integral_channel(out_integrals, channel_idx, shift_amount)


@cc.export(
    "get_mean_by_integral_logic",
    "float64(float64[:], int64, int64, int64)",
)
def get_mean_by_integral_logic_export(s_abs, start, end, t_head):
    return get_mean_by_integral_logic(s_abs, start, end, t_head)


@cc.export("fill_original_data", "float64(float64[:], float64[:], int64)")
def fill_original_data_export(out_integral, increments, head):
    return fill_original_data(out_integral, increments, head)


@cc.export(
    "fill_original_multi",
    "float64[:](float64[:,:], float64[:], float64[:,:], int64[:], int64[:])",
)
def fill_original_multi_export(out_integrals, currents, increments_batch, heads, channels):
    return fill_original_multi(out_integrals, currents, increments_batch, heads, channels)


@cc.export(
    "compute_extremums_inplace_circular",
    "void(float64[:], float64[:], float64[:], int64[:], int64, int64)",
)
def compute_extremums_inplace_circular_export(
    new_data, values, extremas, semi_status, head, total_count
):
    compute_extremums_inplace_circular(
        new_data, values, extremas, semi_status, head, total_count
    )


@cc.export(
    "compute_super_extremas_realtime_circular",
    "void(float64[:], float64[:], float64[:], float64[:], int64, int64, int64, int64)",
)
def compute_super_extremas_realtime_circular_export(
    values,
    extremas,
    super_extremas,
    full_extremas,
    head,
    batch_size,
    window,
    total_count,
):
    compute_super_extremas_realtime_circular(
        values,
        extremas,
        super_extremas,
        full_extremas,
        head,
        batch_size,
        window,
        total_count,
    )


@cc.export("write_abs_time_circular", "void(float64[:], int64, int64, int64)")
def write_abs_time_circular_export(abs_time, head, batch_size, total_count):
    write_abs_time_circular(abs_time, head, batch_size, total_count)


if __name__ == "__main__":
    cc.compile()
