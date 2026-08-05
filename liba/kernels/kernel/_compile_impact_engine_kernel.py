"""AOT export specification for price-impact inversion."""

from numba.pycc import CC

from .impact_engine_kernel import forward, inverse, inverse2


cc = CC("impact_engine_kernel_core")


@cc.export(
    "inverse",
    "float64[:,:](float64[:,:], float64[:], int64[:], types.Tuple((float64, float64)))",
)
def inverse_export(time_square, price_moves, channel_indices, beta_range):
    return inverse(time_square, price_moves, channel_indices, beta_range)


@cc.export(
    "inverse2",
    "float64[:,:](float64[:,:], float64[:], int64[:], types.Tuple((float64, float64)), boolean, int64)",
)
def inverse2_export(
    time_square,
    price_moves,
    channel_indices,
    beta_range,
    consider_offset,
    steps,
):
    return inverse2(
        time_square,
        price_moves,
        channel_indices,
        beta_range,
        consider_offset,
        steps,
    )


@cc.export("forward", "float64[:,:](float64[:,:], float64[:,:])")
def forward_export(active_net_flow, params):
    return forward(active_net_flow, params)


if __name__ == "__main__":
    cc.compile()
