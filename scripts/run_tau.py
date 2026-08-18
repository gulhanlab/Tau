#!/usr/bin/env python3
"""Thin alias for ``tau run``.

Historically the CLI lived here as ``python -m scripts.run_tau ...``. The real
implementation now lives in :mod:`tau.cli`. This module forwards all arguments
to the ``run`` subcommand so existing invocations keep working:

    python -m scripts.run_tau --sample S --vcf ... --cnv ... --purity 0.8 ...

is equivalent to:

    tau run --sample S --vcf ... --cnv ... --purity 0.8 ...
"""

import sys

from tau.cli import main as _cli_main


def main():
    # Forward to `tau run`, preserving the legacy flat argument interface.
    _cli_main(["run", *sys.argv[1:]])


if __name__ == "__main__":
    main()
