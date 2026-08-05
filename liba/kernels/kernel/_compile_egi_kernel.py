"""AOT export specification for the EGI kernel."""

from numba.pycc import CC

from .egi_kernel import egi_update


cc = CC("egi_kernel_core")


@cc.export(
    "egi_update",
    "void(float64[:,:], float64[:,:], i8, i8, f8, i8, i8, f8)",
)
def egi_update_export(
    data,
    integrals,
    t_head,
    increment,
    global_baseline,
    min_window,
    max_window,
    minimal_tick,
):
    egi_update(
        data,
        integrals,
        t_head,
        increment,
        global_baseline,
        min_window,
        max_window,
        minimal_tick,
    )


if __name__ == "__main__":
    cc.compile()
