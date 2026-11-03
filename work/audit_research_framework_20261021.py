"""Read existing P1 evidence only; no model loading, inference, training or dataset edits."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/research_reset_20261021"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_table = ROOT / "docs/thesis/chapter3_20261019/tables/table_3_01_validation_all_11.csv"
    old = {str(ROOT / row["run"]): row for row in csv.DictReader(old_table.open())}
    sources = {str(old_table): digest(old_table)}
    rows, confusion_results = [], {}
    for path in sorted((ROOT / "outputs/P1").rglob("best_val_metrics.json")):
        run = path.parent
        config, status, best = [read(p) for p in (run / "config.json", run / "status.json", path)]
        for p in (run / "config.json", run / "status.json", path):
            sources[str(p)] = digest(p)
        family = run.relative_to(ROOT / "outputs/P1").parts[0]
        if "reference_adapter" in run.name:
            initialization = "frozen_base_adapter_after_F0_40"
        elif family in ("material_categorical", "sampler_aligned", "categorical_rollout"):
            initialization = "shared_Direct40_warmstart_and_prior"
        else:
            initialization = "scratch"
        m = best["metrics"]
        actual_updates = int(old[str(run)]["stage_optimizer_updates"]) if str(run) in old else status["global_step"]
        row = dict(run=str(run), family=family, name=run.name,
                   mode=config.get("arm", config.get("mode", config.get("method"))),
                   initialization=initialization, training_seed=config["seed"],
                   configured_epochs=config["epochs"], completed_epochs=status["last_completed_epoch"],
                   status=status["status"], planned_batches_executed=status["global_step"],
                   optimizer_updates=actual_updates, best_epoch=best["epoch"],
                   best_miou=m["miou_4class"], best_fg_miou=m["miou_foreground"],
                   best_monolayer_iou=m["iou_monolayer"], best_fewlayer_iou=m["iou_fewlayer"],
                   best_multilayer_iou=m["iou_multilayer"], best_boundary_fg=m["boundary_f1_foreground"],
                   sampling_steps=(1 if config.get("mode", config.get("method")) in ("N0", "S0", "direct")
                                   else config.get("sampling_steps", config.get("iterations", 5))),
                   exact40_miou=None, exact40_boundary_fg=None)
        epoch40 = run / "validation_epoch40.json"
        if epoch40.exists():
            v = read(epoch40)
            row.update(exact40_miou=v["miou_4class"], exact40_boundary_fg=v["boundary_f1_foreground"])
            sources[str(epoch40)] = digest(epoch40)
        rows.append(row)
        matrix_file = run / "best_val_confusion.npz"
        if family == "material_categorical" and matrix_file.exists():
            with np.load(matrix_file) as data:
                matrix = data["global_matrix"]
                assert np.array_equal(data["per_image"].sum(0), matrix)
                iou = matrix.diagonal() / (matrix.sum(1) + matrix.sum(0) - matrix.diagonal())
                assert abs(iou.mean() - m["miou_4class"]) < 1e-12
                pairs = [(int(matrix[i,j]), i, j) for i in range(4) for j in range(4) if i != j]
                names = ("background", "monolayer", "fewlayer", "multilayer")
                confusion_results[config["mode"]] = dict(
                    matrix=matrix.tolist(), best_epoch=best["epoch"],
                    recall_per_true_class=(matrix.diagonal()/matrix.sum(1)).tolist(),
                    errors_by_true_class=(matrix.sum(1)-matrix.diagonal()).tolist(),
                    largest_error_pairs=[dict(true=names[i], predicted=names[j], pixels=n)
                                         for n,i,j in sorted(pairs, reverse=True)[:6]],
                    metrics=m,
                    per_image_few_to_mono=sorted(
                        [dict(id=str(ident), pixels=int(v[2,1])) for ident, v in zip(data["ids"], data["per_image"])],
                        key=lambda x:x["pixels"], reverse=True))
            sources[str(matrix_file)] = digest(matrix_file)
    assert len(rows) == 24, len(rows)
    assert all(row["training_seed"] == 42 for row in rows)
    with (OUT / "experiment_registry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="24 unique P1 run directories with full-val evidence; excludes P0, engineering, reused S0 and inference reruns",
                  count=len(rows), initialization_counts=dict(Counter(x["initialization"] for x in rows)),
                  configured_vs_completed_note="Five runs configured100 stopped at40; best differs from exact40 gates",
                  optimizer_updates_total=sum(x["optimizer_updates"] for x in rows),
                  runs=rows, material_confusions=confusion_results,
                  source_sha256=sources,
                  no_new_model_execution=True, no_test_evaluation=True)
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("runs", "source_sha256", "material_confusions")}, indent=2))
    for mode, entry in confusion_results.items():
        print(mode, entry["largest_error_pairs"][:3])
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
