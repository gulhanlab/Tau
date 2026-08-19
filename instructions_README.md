# Tau — see README.md

This file is retained only as a redirect. The authoritative documentation
(installation, the zero-input `tau demo`, the `run_sample()` API, the `tau` CLI,
input file formats, and development instructions) now lives in
[`README.md`](README.md).

Quick pointers:

- **Install:** `pixi install` (provides SageMath). `pip install -e .` does **not**
  install Sage and will not be able to run timing.
- **Try it with no data:** `pixi run demo` (or `tau demo`).
- **Run on your data:** `tau run --sample S --snv_table ... --cn_table ... --purity 0.8`.
- **Python API:** `from tau import run_sample`.
- **Tests:** `pixi run test` (or `pytest tests/`).
