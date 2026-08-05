"""Numerical kernels with an optional Numba AOT backend."""

from __future__ import annotations

import os
from importlib import import_module

from . import egi_kernel, impact_engine_kernel, integrals_kernel, pulsa_data_kernel
from .egi_kernel import EgiDataChannel, EgiIntegralChannel
from .pulsa_data_kernel import PulsaDataChannel, PulsaInputChannel


_AOT_EXPORTS = {
    "pulsa_data_kernel": ("render_liquidity_cube_integrated",),
    "integrals_kernel": (
        "n_get_next_head",
        "update_integral",
        "update_integral_multi",
        "update_integral_normal",
        "update_integral_abs",
        "shift_integral_channel",
        "get_mean_by_integral_logic",
        "fill_original_data",
        "fill_original_multi",
        "compute_extremums_inplace_circular",
        "compute_super_extremas_realtime_circular",
        "write_abs_time_circular",
    ),
    "impact_engine_kernel": ("inverse", "inverse2", "forward"),
    "egi_kernel": ("egi_update",),
}


def _activate_aot_backend() -> tuple[bool, str | None]:
    """Replace JIT dispatchers with compatible native CPython extensions."""
    preference = os.environ.get("VOL_IMPULSE_AOT", "auto").strip().lower()
    if preference in {"0", "false", "off", "python", "jit"}:
        return False, None

    python_modules = {
        "pulsa_data_kernel": pulsa_data_kernel,
        "integrals_kernel": integrals_kernel,
        "impact_engine_kernel": impact_engine_kernel,
        "egi_kernel": egi_kernel,
    }
    try:
        for module_name, exports in _AOT_EXPORTS.items():
            compiled = import_module(f".{module_name}_core", __name__)
            target = python_modules[module_name]
            for export_name in exports:
                setattr(target, export_name, getattr(compiled, export_name))
    except (ImportError, AttributeError) as error:
        if preference in {"1", "true", "on", "aot", "required"}:
            raise RuntimeError(
                "VOL_IMPULSE_AOT requires compiled extensions. Run "
                "`python build.py --force` first."
            ) from error
        return False, str(error)
    return True, None


AOT_ENABLED, AOT_LOAD_ERROR = _activate_aot_backend()


def backend_status() -> dict[str, str | bool | None]:
    """Return a serializable description of the active numerical backend."""
    return {
        "backend": "numba-aot" if AOT_ENABLED else "numba-jit",
        "aot_enabled": AOT_ENABLED,
        "load_error": AOT_LOAD_ERROR,
    }


__all__ = [
    "AOT_ENABLED",
    "EgiDataChannel",
    "EgiIntegralChannel",
    "PulsaDataChannel",
    "PulsaInputChannel",
    "backend_status",
    "egi_kernel",
    "impact_engine_kernel",
    "integrals_kernel",
    "pulsa_data_kernel",
]
