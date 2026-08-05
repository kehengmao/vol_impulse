"""Build every numerical kernel as a native CPython extension."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all Numba AOT extensions in place."
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing extensions")
    parser.add_argument("--clean", action="store_true", help="remove generated extensions")
    parser.add_argument(
        "--no-verify", action="store_true", help="skip clean-process backend verification"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["VOL_IMPULSE_AOT"] = "0"
    from liba.kernels.kernel._compile import clean_extensions, compile_all

    if args.clean:
        removed = clean_extensions()
        print(f"Removed {len(removed)} generated extension(s).")
        if not args.force:
            return 0

    built = compile_all(force=args.force)
    print(f"Built {len(built)} extension(s).")

    if not args.no_verify:
        environment = os.environ.copy()
        environment["VOL_IMPULSE_AOT"] = "1"
        command = [
            sys.executable,
            "-c",
            (
                "import json; from liba import backend_status; "
                "print(json.dumps(backend_status()))"
            ),
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        status = json.loads(result.stdout.strip().splitlines()[-1])
        if not status.get("aot_enabled"):
            raise RuntimeError(f"AOT verification failed: {status}")
        print("AOT verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
