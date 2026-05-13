import os
import csv
import itertools
import numpy as np

from benchmark_runner import run_one_mission, BENCHMARK_SEEDS
from agents.cabbage_agent import CabbageAgent
from agents.hybrid_agent import HybridAgent
from core.checkpoint import CheckpointManager
from core.tuning_config import runtime_config


SEARCH_SPACE = {
    "TURN_COST_WEIGHT": [0.2, 0.3, 0.4],
    "UNKNOWN_COST_ALLOW": [0.2, 0.3, 0.5],
    "DYNAMIC_NEAR_COST": [1.0, 2.0, 3.0],
    "REPLAN_INTERVAL": [5, 8, 10],
}


def config_product(space):
    keys = list(space.keys())
    values = [space[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def score_results(rows):
    success_rate = np.mean([r["success"] for r in rows])
    collect_rate = np.mean([r["collect_rate"] for r in rows])
    energy_per_cabbage = np.mean([r["energy_per_cabbage"] for r in rows])
    turns = np.mean([r["turns"] for r in rows])
    overlap = np.mean([r["overlap_rate"] for r in rows])

    score = (
        success_rate * 1000
        + collect_rate * 300
        - energy_per_cabbage * 20
        - turns * 0.5
        - overlap * 100
    )

    return score, {
        "success_rate": success_rate,
        "collect_rate": collect_rate,
        "energy_per_cabbage": energy_per_cabbage,
        "turns": turns,
        "overlap": overlap,
    }


def save_tuning_results(rows, path="logs/auto_tune_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Auto-tune results saved: {path}")


def main():
    local_agent = CabbageAgent()

    ckpt = CheckpointManager(
        k_best=3,
        project_name="Cab2"
    )

    ckpt.load_checkpoint(local_agent)

    all_rows = []
    best_score = -1e18
    best_config = None
    best_metrics = None

    configs = list(config_product(SEARCH_SPACE))

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 60)
        print(f"CONFIG {i + 1}/{len(configs)}")
        print(cfg)

        runtime_config.update(cfg)

        agent = HybridAgent(
            local_agent=local_agent,
            robot_id="robot_1"
        )

        rows = []

        for seed in BENCHMARK_SEEDS:
            result = run_one_mission(agent, seed)
            rows.append(result)

        score, metrics = score_results(rows)

        print("score:", score)
        print("metrics:", metrics)

        flat_row = {
            "config_id": i,
            "score": score,
            **cfg,
            **metrics,
        }

        all_rows.append(flat_row)

        if score > best_score:
            best_score = score
            best_config = cfg
            best_metrics = metrics

            print("🔥 NEW BEST")
            print(best_config)
            print(best_metrics)

    save_tuning_results(all_rows)

    print("\n========== BEST CONFIG ==========")
    print(best_config)
    print("score:", best_score)
    print(best_metrics)


if __name__ == "__main__":
    main()