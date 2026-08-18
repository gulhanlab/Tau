"""tau.simulate — config-driven simulation COHORT generator (SKETCH / interface).

Shippable tool (`tau simulate`) so users can design their own simulation runs to stress-test Tau on new
scenarios. Wraps the single-genome logic in `tau.simulation` (which built the paper's v8 cohort) with a
customizable config, and writes each sim in the standard per-sample layout (+ `{SIM}.truth.json`) so the SAME
`tau aggregate` / `tau benchmark` evaluate it. Completes the loop: simulate → run → aggregate → benchmark.
STATUS: interface only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    """Knobs a user customizes to define a simulation cohort (copy default_config() or a YAML and edit)."""
    n_samples: int                                # cohort size (no default -> must be set)
    seed_base: int = 6000                         # per-sample seed = seed_base + i (reproducible)
    events: list = field(default_factory=list)    # scenarios drawn per genome, e.g.
                                                  #   [{"type":"WGD","time_frac":[0.2,0.9]},   # uniform range
                                                  #    {"type":"PGD","n":[0,2]},               # 0-2 PGDs
                                                  #    {"type":"chrom_specific","n":[0,5]}]
    cn_states: dict = field(default_factory=dict)  # CN-state mix per genome ((major,minor) -> weight)
    purity: float | list = 0.7                    # fixed purity or [lo, hi] range
    mutation_burden: int | list = 2000            # SNVs per genome (controls ess); fixed or range
    n_bootstraps: int = 1                         # bootstrap replicates per sample
    run_tau: bool = True                          # recover timing after generating truth (writes genome pkl)
    emit_tables: bool = True                      # also emit per-sample tau_* tables (tau.aggregate.export)
    tumor_type: str = "SIM"                       # label for tumor_type column / dir grouping
    extra: dict = field(default_factory=dict)     # scenario-specific overrides


def default_config() -> SimConfig:
    """A reference config that reproduces the paper's v8 benchmark cohort (WGD+PGD combos, seed_base=6000,
    purity≈0.5). Users copy + edit this (or a YAML) as the starting point."""
    raise NotImplementedError


def simulate_cohort(config: SimConfig, out_dir: Path) -> Path:
    """Generate a simulation cohort into `out_dir` (the sim analog of pcawg_run/samples/).

    For each of `config.n_samples`: draw a scenario, build a synthetic genome with known truth via
    `tau.simulation` (→ constraint matrix → Poisson N_counts), write:
      {SIM}/{SIM}.genome.pkl.gz         (+ recovered timing if run_tau)
      {SIM}/{SIM}.truth.json            (ground-truth events/segments/routes)
      {SIM}/{SIM}.tau_*.{tsv,parquet}   (if emit_tables — via tau.aggregate.export)
    Deterministic given `seed_base`. Returns `out_dir`. Then `tau aggregate` + `tau benchmark` evaluate it.
    """
    raise NotImplementedError
