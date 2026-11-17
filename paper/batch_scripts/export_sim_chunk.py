"""Export one chunk of SIMULATION genome pkls → per-sim composite tables (tau_segments, tau_routes,
tau_multiplicities, tau_gains, tau_samples). Reshape only, NO re-timing — the sim pkls are already
timed tau.core.Genome objects (same structure as PCAWG).
usage: python export_sim_chunk.py <task_id> <chunk_size>   (SLURM array)
-> per-sim tables under paper/data_tables/sim_stage/
"""
import sys, glob
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from tau.aggregate.export import export_tables_from_pkl  # noqa: E402

SIM = R / "dev/benchmarking/output_v8"
STAGE = R / "paper/data_tables/sim_stage"


def main(task_id, chunk):
    pkls = sorted(glob.glob(str(SIM / "SIM_*" / "*.genome.pkl.gz")))
    start, end = (task_id - 1) * chunk, min(task_id * chunk, len(pkls))
    if start >= len(pkls):
        print(f"[task {task_id}] nothing to do"); return
    STAGE.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in pkls[start:end]:
        sid = Path(p).name.replace(".genome.pkl.gz", "")
        try:
            export_tables_from_pkl(p, STAGE)
            n += 1
        except Exception as e:
            print(f"  {sid}: export failed {e}", flush=True)
    print(f"[task {task_id}] pkls[{start}:{end}]  exported={n}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
