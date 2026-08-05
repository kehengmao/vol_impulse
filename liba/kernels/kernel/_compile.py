"""Unified in-place builder for all Numba ``pycc`` kernels."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


KERNEL_DIR = Path(__file__).resolve().parent
COMPILER_MODULES = (
    "_compile_pulsa_data_kernel",
    "_compile_integrals_kernel",
    "_compile_impact_engine_kernel",
    "_compile_egi_kernel",
)


def _extension_files(core_name: str) -> list[Path]:
    return [
        path
        for path in KERNEL_DIR.glob(f"{core_name}*")
        if path.suffix.lower() in {".pyd", ".so", ".dll", ".dylib"}
    ]


def clean_extensions() -> list[Path]:
    """Delete only generated ``*_kernel_core`` native extensions."""
    removed = []
    for path in KERNEL_DIR.glob("*_kernel_core*"):
        if path.suffix.lower() in {".pyd", ".so", ".dll", ".dylib"}:
            path.unlink()
            removed.append(path)
    return removed


def compile_all(force: bool = False) -> list[Path]:
    """Compile every registered kernel and return the generated files."""
    previous_directory = Path.cwd()
    built = []
    try:
        os.chdir(KERNEL_DIR)
        for module_name in COMPILER_MODULES:
            module = importlib.import_module(f"{__package__}.{module_name}")
            compiler = module.cc
            existing = _extension_files(compiler.name)
            if existing and not force:
                print(f"Using existing extension: {existing[0].name}")
                continue
            if force:
                for path in existing:
                    path.unlink()
            print(f"Compiling {compiler.name} ...")
            compiler.compile()
            generated = _extension_files(compiler.name)
            if not generated:
                raise RuntimeError(f"No extension was generated for {compiler.name}")
            built.extend(generated)
            print(f"Built {generated[0].name}")
    finally:
        os.chdir(previous_directory)
    return built


if __name__ == "__main__":
    compile_all()
