"""AOT export specification for liquidity-cube construction."""

from numba.pycc import CC

from .pulsa_data_kernel import render_liquidity_cube_integrated


cc = CC("pulsa_data_kernel_core")


@cc.export("render_liquidity_cube_integrated", "f8[:,:,:](f8[:,:], f8)")
def render_liquidity_cube_export(data, iceberg_limit_factor):
    return render_liquidity_cube_integrated(data, iceberg_limit_factor)


if __name__ == "__main__":
    cc.compile()
