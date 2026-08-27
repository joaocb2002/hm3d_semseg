"""Static, human-readable reports built from authoritative run artifacts."""

# HTML/CSS/JavaScript template lines remain intact for readability.
# ruff: noqa: E501

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from hm3d_semseg.training.report_plots import save_report_plots
from hm3d_semseg.training.reporting import summarize_training_metrics
from hm3d_semseg.utils.hashing import (
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    sha256_file,
)

EPOCH_COLUMNS = (
    "epoch",
    "training_cross_entropy",
    "development_cross_entropy",
    "known_class_miou",
    "objectnav_six_miou",
    "overall_pixel_accuracy",
    "mean_class_recall",
    "scene_macro_mean_miou",
    "nll",
    "multiclass_brier",
    "ece",
)


def generate_training_report(run: Path) -> Dict[str, Any]:
    """Build or backfill one self-contained static run report."""
    run = run.resolve()
    metrics_path = run / "metrics.jsonl"
    if not metrics_path.is_file():
        raise ValueError(f"Training metrics do not exist: {metrics_path}")
    report_root = run / "report"
    tables_root = report_root / "tables"
    plots_root = report_root / "plots"
    tables_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    run_summary = _read_json(run / "summary.json") or {}
    provenance = _read_json(run / "provenance.json") or {}
    parameters = _read_json(run / "parameter_counts.json") or {}
    resolved = _read_yaml(run / "resolved_config.yaml") or {}
    metrics_summary = summarize_training_metrics(metrics_path)
    evaluations = _load_evaluations(run)
    epoch_rows = _epoch_rows(metrics_path, evaluations)
    best_evaluation = _best_evaluation(evaluations)
    checkpoint_rows = _checkpoint_rows(evaluations)

    table_paths: List[Path] = []
    table_paths.append(_write_csv(tables_root / "epoch_metrics.csv", epoch_rows))
    table_paths.append(_write_csv(tables_root / "checkpoint_comparison.csv", checkpoint_rows))
    per_class_rows: List[Dict[str, Any]] = []
    objectnav_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    if best_evaluation is not None:
        best_epoch, best = best_evaluation
        per_class_rows = [
            {
                "epoch": best_epoch + 1,
                "class_id": item["id"],
                "class_name": item["name"],
                "support": item["support"],
                "predicted": item["predicted"],
                "iou": item["iou"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
            }
            for item in best["global"]["per_class"]
        ]
        objectnav_rows = [
            {"epoch": best_epoch + 1, "goal": goal, **values}
            for goal, values in best["global"]["objectnav_six"].items()
        ]
        scene_rows = [
            {
                "epoch": best_epoch + 1,
                "scene_id": scene_id,
                "known_class_miou": value,
            }
            for scene_id, value in sorted(
                best["scene_macro"]["per_scene_known_class_miou"].items()
            )
        ]
        confusion_rows = _top_confusions(best, best_epoch)
    table_paths.extend(
        [
            _write_csv(tables_root / "per_class_best.csv", per_class_rows),
            _write_csv(tables_root / "objectnav_six_best.csv", objectnav_rows),
            _write_csv(tables_root / "per_scene_best.csv", scene_rows),
            _write_csv(tables_root / "top_confusions_best.csv", confusion_rows),
        ]
    )
    plot_paths = save_report_plots(metrics_path, evaluations, plots_root)
    qualitative = _qualitative_history(run)
    facts, warnings = _facts_and_warnings(
        run_summary,
        provenance,
        resolved,
        metrics_summary,
        evaluations,
        qualitative,
    )
    report_summary = {
        "schema_version": "1.0",
        "run": str(run),
        "run_name": run.name,
        "facts": facts,
        "warnings": warnings,
        "epoch_metrics": epoch_rows,
        "checkpoint_comparison": checkpoint_rows,
        "train_subset_evaluation": run_summary.get("train_subset_evaluation"),
        "best_epoch": best_evaluation[0] + 1 if best_evaluation else None,
        "best_per_class": per_class_rows,
        "best_objectnav_six": objectnav_rows,
        "best_per_scene": scene_rows,
        "top_confusions": confusion_rows,
        "qualitative": qualitative,
        "plots": [str(path.relative_to(report_root)) for path in plot_paths],
        "tables": [str(path.relative_to(report_root)) for path in table_paths],
        "raw_artifacts_preserved": True,
    }
    atomic_write_json(report_root / "summary.json", report_summary)
    atomic_write_text(
        report_root / "summary.md",
        _markdown_summary(report_summary, metrics_summary),
    )
    atomic_write_text(
        report_root / "index.html",
        _html_report(report_summary, metrics_summary, parameters, provenance),
    )
    generated = [
        report_root / "summary.json",
        report_root / "summary.md",
        report_root / "index.html",
        *table_paths,
        *plot_paths,
    ]
    manifest = {
        "schema_version": "1.0",
        "derived_from": {
            "metrics_jsonl": str(metrics_path),
            "metrics_jsonl_sha256": sha256_file(metrics_path),
            "evaluation_reports": [
                {
                    "path": str(
                        (
                            run / f"evaluation-epoch-{epoch:03d}" / "summary.json"
                        ).resolve()
                    ),
                    "sha256": sha256_file(
                        run / f"evaluation-epoch-{epoch:03d}" / "summary.json"
                    ),
                }
                for epoch, _ in evaluations
            ],
        },
        "generated": [
            {
                "path": str(path.relative_to(report_root)),
                "sha256": sha256_file(path),
            }
            for path in generated
            if path.is_file()
        ],
    }
    atomic_write_json(report_root / "report_manifest.json", manifest)
    return {
        "run": str(run),
        "report": str(report_root / "index.html"),
        "summary": str(report_root / "summary.md"),
        "warnings": warnings,
        "plots": len(plot_paths),
        "tables": len(table_paths),
        "backfilled": True,
    }


def compare_training_runs(runs: Sequence[Path], output: Path) -> Dict[str, Any]:
    """Create an objective held-out comparison for two or more training runs."""
    if len(runs) < 2:
        raise ValueError("At least two run directories are required")
    resolved_runs = [path.resolve() for path in runs]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_data: List[
        Tuple[
            Path,
            List[Tuple[int, Dict[str, Any]]],
            Tuple[int, Dict[str, Any]],
            Dict[str, Any],
        ]
    ] = []
    for run in resolved_runs:
        generate_training_report(run)
        evaluations = _load_evaluations(run)
        best_selection = _best_evaluation(evaluations)
        if best_selection is None:
            raise ValueError(f"Run has no development evaluations: {run}")
        provenance = _read_json(run / "provenance.json") or {}
        run_data.append((run, evaluations, best_selection, provenance))

    comparability = _comparability(run_data)
    overview = []
    for run, evaluations, (epoch, best_report), provenance in run_data:
        metric_summary = summarize_training_metrics(run / "metrics.jsonl")
        optimization = metric_summary["training"]["optimization"]
        overview.append(
            {
                "run": run.name,
                "path": str(run),
                "best_epoch": epoch + 1,
                "best_known_class_miou": best_report["global"]["known_class_miou"],
                "best_objectnav_six_miou": best_report["global"][
                    "objectnav_six_miou"
                ],
                "best_overall_pixel_accuracy": best_report["global"][
                    "overall_pixel_accuracy"
                ],
                "best_scene_macro_mean_miou": best_report["scene_macro"]["mean"],
                "best_development_cross_entropy": best_report[
                    "mean_cross_entropy_loss"
                ],
                "epochs_evaluated": len(evaluations),
                "overall_samples_per_second": optimization.get(
                    "overall_samples_per_second"
                ),
                "total_step_hours": (
                    float(optimization["total_step_seconds"]) / 3600.0
                    if optimization.get("total_step_seconds") is not None
                    else None
                ),
                "peak_gpu_memory_gib": (
                    float(optimization["peak_gpu_memory_bytes"]) / (1024**3)
                    if optimization.get("peak_gpu_memory_bytes") is not None
                    else None
                ),
                "git_commit": provenance.get("hm3d_semseg_commit"),
            }
        )
    overview_path = _write_csv(output / "run_overview.csv", overview)
    paired = _paired_scene_comparison(run_data) if len(run_data) == 2 else None
    if paired is not None:
        _write_csv(output / "paired_scene_differences.csv", paired["scenes"])
        _write_csv(output / "per_class_differences.csv", paired["classes"])
    plot_paths = _save_comparison_plots(run_data, output)
    qualitative = []
    for run, _, (epoch, _), _ in run_data:
        sheet = (
            run
            / "diagnostics"
            / "training_progress"
            / "qualitative"
            / "development"
            / "contact_sheets"
            / f"epoch_{epoch:03d}.png"
        )
        copied_sheet = None
        if sheet.is_file():
            copied_sheet = (
                output / "qualitative" / f"{run.name}-epoch-{epoch + 1:03d}.png"
            )
            copied_sheet.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sheet, copied_sheet)
        qualitative.append(
            {
                "run": run.name,
                "epoch": epoch + 1,
                "path": (
                    Path(os.path.relpath(copied_sheet, output)).as_posix()
                    if copied_sheet is not None
                    else None
                ),
            }
        )
    comparison = {
        "schema_version": "1.0",
        "runs": overview,
        "comparability": comparability,
        "paired_comparison": paired,
        "qualitative_best_epoch": qualitative,
        "plots": [path.name for path in plot_paths],
        "training_loss_compared": False,
        "reason": (
            "Class-weighted and unweighted training losses are different objectives; "
            "held-out metrics are compared instead."
        ),
    }
    atomic_write_json(output / "summary.json", comparison)
    atomic_write_text(output / "summary.md", _comparison_markdown(comparison))
    atomic_write_text(output / "index.html", _comparison_html(comparison))
    return {
        "output": str(output),
        "report": str(output / "index.html"),
        "overview": str(overview_path),
        "comparable": comparability["comparable"],
        "warnings": comparability["warnings"],
    }


def _load_evaluations(run: Path) -> List[Tuple[int, Dict[str, Any]]]:
    result = []
    pattern = re.compile(r"evaluation-epoch-(\d+)$")
    for path in sorted(run.glob("evaluation-epoch-*/summary.json")):
        match = pattern.match(path.parent.name)
        if match:
            result.append((int(match.group(1)), _read_json(path) or {}))
    return result


def _epoch_rows(
    metrics_path: Path, evaluations: Sequence[Tuple[int, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    train_losses: Dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "train_epoch":
                train_losses[int(item["epoch"])] = float(item["loss"])
    evaluation_by_epoch = dict(evaluations)
    epochs = sorted(set(train_losses) | set(evaluation_by_epoch))
    rows = []
    for epoch in epochs:
        report = evaluation_by_epoch.get(epoch)
        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "training_cross_entropy": train_losses.get(epoch),
        }
        if report is None:
            row.update({key: None for key in EPOCH_COLUMNS[2:]})
        else:
            row.update(
                {
                    "development_cross_entropy": report.get("mean_cross_entropy_loss"),
                    "known_class_miou": report["global"].get("known_class_miou"),
                    "objectnav_six_miou": report["global"].get("objectnav_six_miou"),
                    "overall_pixel_accuracy": report["global"].get("overall_pixel_accuracy"),
                    "mean_class_recall": report["global"].get("mean_class_recall"),
                    "scene_macro_mean_miou": report["scene_macro"].get("mean"),
                    "nll": report["probability_quality"].get("nll"),
                    "multiclass_brier": report["probability_quality"].get("multiclass_brier"),
                    "ece": report["probability_quality"].get("ece"),
                }
            )
        rows.append(row)
    return rows


def _best_evaluation(
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
) -> Optional[Tuple[int, Dict[str, Any]]]:
    valid = [
        item
        for item in evaluations
        if item[1].get("global", {}).get("known_class_miou") is not None
    ]
    return (
        max(valid, key=lambda item: float(item[1]["global"]["known_class_miou"]))
        if valid
        else None
    )


def _checkpoint_rows(evaluations: Sequence[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not evaluations:
        return []
    loss_candidates = [
        item for item in evaluations if item[1].get("mean_cross_entropy_loss") is not None
    ]
    best = _best_evaluation(evaluations)
    selected: List[Tuple[str, Tuple[int, Dict[str, Any]]]] = []
    if loss_candidates:
        selected.append(
            (
                "minimum development cross-entropy",
                min(
                    loss_candidates,
                    key=lambda item: float(item[1]["mean_cross_entropy_loss"]),
                ),
            )
        )
    if best is not None:
        selected.append(("best known-class mIoU (checkpoints/best)", best))
    selected.append(("final (checkpoints/last)", evaluations[-1]))
    rows = []
    for label, (epoch, report) in selected:
        rows.append(
            {
                "role": label,
                "epoch": epoch + 1,
                "development_cross_entropy": report.get("mean_cross_entropy_loss"),
                "known_class_miou": report["global"].get("known_class_miou"),
                "objectnav_six_miou": report["global"].get("objectnav_six_miou"),
                "overall_pixel_accuracy": report["global"].get("overall_pixel_accuracy"),
                "scene_macro_mean_miou": report["scene_macro"].get("mean"),
                "nll": report["probability_quality"].get("nll"),
                "multiclass_brier": report["probability_quality"].get("multiclass_brier"),
                "ece": report["probability_quality"].get("ece"),
            }
        )
    return rows


def _facts_and_warnings(
    summary: Dict[str, Any],
    provenance: Dict[str, Any],
    resolved: Dict[str, Any],
    metrics: Dict[str, Any],
    evaluations: Sequence[Tuple[int, Dict[str, Any]]],
    qualitative: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    facts = [
        f"Recorded {metrics['training']['epochs_recorded']} training epochs and "
        f"{metrics['training']['optimizer_steps_recorded']:,} optimizer steps.",
        f"Training subset: {summary.get('train_samples', 'unknown')} samples across "
        f"{summary.get('train_scenes', 'unknown')} scenes.",
    ]
    development_samples = summary.get("development_samples")
    if development_samples is not None:
        facts.append(
            f"Development evaluation: {development_samples} samples across "
            f"{summary.get('development_scenes', 'unknown')} held-out scenes."
        )
    weighting = summary.get("class_weighting") or resolved.get("training", {}).get(
        "class_weighting", "unknown"
    )
    facts.append(f"Training loss class weighting: {weighting}.")
    subset = summary.get("train_subset_evaluation")
    if subset is not None:
        facts.append(
            "Selected training-subset memorization: "
            f"pixel accuracy {_format(subset.get('overall_pixel_accuracy'))}, "
            f"known-class mIoU {_format(subset.get('known_class_miou'))}."
        )
    commit = provenance.get("hm3d_semseg_commit")
    if commit:
        facts.append(f"Repository commit: {commit}.")
    warnings = []
    gradient = metrics["training"]["gradient_norm"]
    if gradient.get("nonfinite_count", 0):
        warnings.append(
            f"Observed {gradient['nonfinite_count']:,} non-finite gradient-norm records "
            f"({gradient['nonfinite_fraction']:.3%} of optimizer-step records)."
        )
    skipped_steps = metrics["training"]["optimization"].get(
        "optimizer_steps_skipped", 0
    )
    if skipped_steps:
        warnings.append(
            f"AMP skipped {skipped_steps:,} optimizer updates after overflow detection."
        )
    if not evaluations:
        warnings.append(
            "No development evaluation exists; this run cannot measure generalization "
            "or select a checkpoint by held-out mIoU."
        )
    else:
        losses = [
            float(report["mean_cross_entropy_loss"])
            for _, report in evaluations
            if report.get("mean_cross_entropy_loss") is not None
        ]
        if losses and losses[-1] > min(losses) * 1.1:
            increase = (losses[-1] / min(losses) - 1.0) * 100.0
            warnings.append(
                "Final development cross-entropy is "
                f"{increase:.1f}% above its minimum; probability generalization worsened."
            )
    if not qualitative["train"]:
        warnings.append(
            "No fixed-set qualitative history is available. This is expected for a run "
            "created before the reporting upgrade."
        )
    if evaluations and not qualitative["development"]:
        warnings.append(
            "No development qualitative history is available. Existing scalar and "
            "confusion metrics remain valid."
        )
    return facts, warnings


def _qualitative_history(run: Path) -> Dict[str, List[Dict[str, Any]]]:
    root = run / "diagnostics" / "training_progress" / "qualitative"
    result: Dict[str, List[Dict[str, Any]]] = {"train": [], "development": []}
    for split in result:
        for path in sorted((root / split / "contact_sheets").glob("epoch_*.png")):
            match = re.search(r"epoch_(\d+)", path.stem)
            if match:
                result[split].append(
                    {
                        "epoch": int(match.group(1)) + 1,
                        "path": Path(os.path.relpath(path, run / "report")).as_posix(),
                    }
                )
    return result


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    fieldnames = list(rows[0]) if rows else []
    buffer = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return path


def _markdown_summary(report: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    lines = [
        f"# Training report: {report['run_name']}",
        "",
        "This is a derived human-facing report. `../metrics.jsonl` and the per-epoch "
        "evaluation JSON files remain authoritative.",
        "",
        "## Run facts",
        "",
    ]
    lines.extend(f"- {value}" for value in report["facts"])
    lines.extend(["", "## Attention", ""])
    lines.extend(
        [f"- {value}" for value in report["warnings"]]
        or ["- No automatic warning was triggered."]
    )
    lines.extend(["", "## Checkpoint comparison", ""])
    if report["checkpoint_comparison"]:
        lines.append(_markdown_table(report["checkpoint_comparison"]))
    else:
        train = metrics["training"]["epoch_cross_entropy"]
        lines.append(
            "No development checkpoint comparison is available. Training epoch "
            f"cross-entropy ended at {_format(train.get('final'))}."
        )
    lines.extend(
        [
            "",
            "## Main files",
            "",
            "- Open [the interactive static report](index.html).",
            "- Inspect [epoch metrics](tables/epoch_metrics.csv).",
            "- Inspect [best-epoch class metrics](tables/per_class_best.csv).",
            "- Inspect the original [append-only metric log](../metrics.jsonl).",
            "",
        ]
    )
    return "\n".join(lines)


def _html_report(
    report: Dict[str, Any],
    metrics: Dict[str, Any],
    parameters: Dict[str, Any],
    provenance: Dict[str, Any],
) -> str:
    plots = "".join(
        f'<figure><img src="{html.escape(path)}" loading="lazy">'
        f"<figcaption>{html.escape(Path(path).stem.replace('_', ' '))}</figcaption></figure>"
        for path in report["plots"]
    )
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in report["warnings"])
    facts = "".join(f"<li>{html.escape(value)}</li>" for value in report["facts"])
    checkpoint_table = _html_table(report["checkpoint_comparison"])
    epoch_table = _html_table(report["epoch_metrics"])
    per_class_table = _html_table(report["best_per_class"])
    confusion_table = _html_table(report["top_confusions"])
    subset_table = _html_table(
        [report["train_subset_evaluation"]]
        if report["train_subset_evaluation"] is not None
        else []
    )
    qualitative = _qualitative_html(report["qualitative"])
    trainable = parameters.get("trainable")
    total = parameters.get("total")
    hardware = provenance.get("device_selection", {}).get("device", "unknown")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Training report — {html.escape(report["run_name"])}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18202a}}
main{{max-width:1280px;margin:auto;padding:28px}} h1,h2{{line-height:1.2}}
.card{{background:white;border:1px solid #d9dfe7;border-radius:10px;padding:18px;margin:16px 0}}
.warning{{background:#fff5db;border-color:#e0b33d}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
figure{{margin:0;background:white;border:1px solid #d9dfe7;border-radius:10px;padding:10px}} img{{max-width:100%;height:auto}}
figcaption{{text-transform:capitalize;color:#52606d}} .table-wrap{{overflow:auto;max-height:650px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:7px;border-bottom:1px solid #dde3ea;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#e8edf3;cursor:pointer}} th:first-child,td:first-child{{text-align:left}}
code{{background:#edf1f5;padding:2px 5px;border-radius:4px}} .muted{{color:#617080}}
.qual img{{width:100%;border:1px solid #ccd3dc}} input[type=range]{{width:min(600px,100%)}}
</style></head><body><main>
<h1>Training report: {html.escape(report["run_name"])}</h1>
<p class="muted">Derived presentation only. Raw JSON/JSONL, checkpoints, provenance, and evaluation artifacts are unchanged.</p>
<section class="card"><h2>At a glance</h2><ul>{facts}</ul>
<p><b>Device:</b> {html.escape(str(hardware))} · <b>Parameters:</b> {_format(trainable)} trainable / {_format(total)} total</p></section>
<section class="card warning"><h2>Attention</h2><ul>{warnings or "<li>No automatic warning was triggered.</li>"}</ul></section>
<section class="card"><h2>Checkpoint comparison</h2><p>The best checkpoint is selected by development known-class mIoU when development data exists.</p><div class="table-wrap">{checkpoint_table}</div></section>
<section class="card"><h2>Selected training-subset diagnostic</h2><p>This measures memorization, not held-out generalization.</p><div class="table-wrap">{subset_table}</div></section>
<section><h2>Curves and diagnostics</h2><div class="grid">{plots}</div></section>
{qualitative}
<section class="card"><h2>Epoch metrics</h2><div class="table-wrap">{epoch_table}</div></section>
<section class="card"><h2>Best-epoch per-class metrics</h2><div class="table-wrap">{per_class_table}</div></section>
<section class="card"><h2>Largest best-epoch confusions</h2><p>Rows are true classes; columns are predicted classes.</p><div class="table-wrap">{confusion_table}</div></section>
<section class="card"><h2>Machine-readable sources</h2><p><a href="../metrics.jsonl">metrics.jsonl</a> · <a href="../metrics_summary.json">metrics_summary.json</a> · <a href="../summary.json">run summary</a> · <a href="../provenance.json">provenance</a> · <a href="tables/epoch_metrics.csv">epoch CSV</a></p></section>
</main><script>
document.querySelectorAll('table.sortable th').forEach((th,i)=>th.onclick=()=>{{const t=th.closest('table'),b=t.tBodies[0],r=[...b.rows],a=th.dataset.asc!=='1';r.sort((x,y)=>{{let A=x.cells[i].dataset.value,B=y.cells[i].dataset.value,nA=Number(A),nB=Number(B);return (Number.isFinite(nA)&&Number.isFinite(nB)?nA-nB:A.localeCompare(B))*(a?1:-1)}});r.forEach(x=>b.appendChild(x));th.dataset.asc=a?'1':'0'}});
</script></body></html>"""


def _qualitative_html(history: Dict[str, List[Dict[str, Any]]]) -> str:
    sections = []
    for split in ("train", "development"):
        items = history[split]
        if not items:
            continue
        identifier = f"qual-{split}"
        encoded = html.escape(json.dumps(items), quote=True)
        sections.append(
            f'<div class="card qual" data-items="{encoded}"><h3>{split.title()} fixed set</h3>'
            f'<label>Epoch <output id="{identifier}-value">{items[-1]["epoch"]}</output></label>'
            f'<input id="{identifier}-slider" type="range" min="0" max="{len(items) - 1}" value="{len(items) - 1}">'
            f'<p><img id="{identifier}-image" src="{html.escape(items[-1]["path"])}"></p></div>'
            f'<script>(()=>{{const data={json.dumps(items)},s=document.getElementById("{identifier}-slider"),i=document.getElementById("{identifier}-image"),o=document.getElementById("{identifier}-value");s.oninput=()=>{{const x=data[Number(s.value)];i.src=x.path;o.value=x.epoch}}}})();</script>'
        )
    if not sections:
        return '<section class="card"><h2>Qualitative history</h2><p>Not available for this run.</p></section>'
    return "<section><h2>Fixed qualitative sets</h2>" + "".join(sections) + "</section>"


def _html_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(name.replace('_', ' '))}</th>" for name in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td data-value="{html.escape(str(value if value is not None else ""))}">'
            f"{html.escape(_format(value))}</td>"
            for value in (row.get(column) for column in columns)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f'<table class="sortable"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'
    )


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(column.replace("_", " ") for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(row.get(column)) for column in columns) + " |" for row in rows
    )
    return "\n".join(lines)


def _comparability(run_data: Sequence[Any]) -> Dict[str, Any]:
    fields = {
        "training_manifest_hash": [
            item[3].get("train_dataset_validation", {}).get("manifest_hash")
            for item in run_data
        ],
        "development_manifest_hash": [
            item[3].get("development_dataset_validation", {}).get("manifest_hash")
            for item in run_data
        ],
        "model_revision": [item[3].get("model_revision") for item in run_data],
        "model_id": [item[3].get("model_id") for item in run_data],
        "seed": [item[3].get("seed") for item in run_data],
        "source_commit": [item[3].get("hm3d_semseg_commit") for item in run_data],
        "training_selection_hash": [
            canonical_hash(item[3].get("training_sample_ids")) for item in run_data
        ],
        "development_selection_hash": [
            canonical_hash(item[3].get("development_sample_ids")) for item in run_data
        ],
        "epochs_evaluated": [len(item[1]) for item in run_data],
    }
    mismatches = [name for name, values in fields.items() if len(set(values)) != 1]
    warnings = [f"Mismatch in {name}." for name in mismatches]
    return {"comparable": not mismatches, "fields": fields, "warnings": warnings}


def _top_confusions(report: Dict[str, Any], epoch: int) -> List[Dict[str, Any]]:
    normalized = report["global"].get("row_normalized_confusion_matrix")
    classes = report["global"].get("per_class", [])
    if normalized is None or len(classes) != 41:
        return []
    rows = []
    for true_id, row in enumerate(normalized):
        for predicted_id, value in enumerate(row):
            if true_id == predicted_id or float(value) <= 0.0:
                continue
            rows.append(
                {
                    "epoch": epoch + 1,
                    "true_id": true_id,
                    "true_class": classes[true_id]["name"],
                    "predicted_id": predicted_id,
                    "predicted_class": classes[predicted_id]["name"],
                    "fraction_of_true_class": float(value),
                }
            )
    return sorted(
        rows, key=lambda item: float(item["fraction_of_true_class"]), reverse=True
    )[:30]


def _paired_scene_comparison(run_data: Sequence[Any]) -> Dict[str, Any]:
    import numpy as np

    first = run_data[0][2][1]
    second = run_data[1][2][1]
    first_scenes = first["scene_macro"]["per_scene_known_class_miou"]
    second_scenes = second["scene_macro"]["per_scene_known_class_miou"]
    common = sorted(set(first_scenes) & set(second_scenes))
    scene_rows = [
        {
            "scene_id": scene,
            "first_miou": first_scenes[scene],
            "second_miou": second_scenes[scene],
            "second_minus_first": float(second_scenes[scene]) - float(first_scenes[scene]),
        }
        for scene in common
        if first_scenes[scene] is not None and second_scenes[scene] is not None
    ]
    differences = np.asarray([item["second_minus_first"] for item in scene_rows])
    rng = np.random.default_rng(2027)
    means = (
        np.asarray(
            [
                np.mean(rng.choice(differences, len(differences), replace=True))
                for _ in range(5000)
            ]
        )
        if len(differences)
        else np.asarray([])
    )
    classes = []
    first_classes = first["global"]["per_class"]
    second_classes = second["global"]["per_class"]
    for left, right in zip(first_classes, second_classes):
        delta = (
            float(right["iou"]) - float(left["iou"])
            if left["iou"] is not None and right["iou"] is not None
            else None
        )
        classes.append(
            {
                "class_id": left["id"],
                "class_name": left["name"],
                "first_iou": left["iou"],
                "second_iou": right["iou"],
                "second_minus_first": delta,
            }
        )
    return {
        "first_run": run_data[0][0].name,
        "second_run": run_data[1][0].name,
        "common_scenes": len(scene_rows),
        "mean_paired_scene_delta": float(differences.mean()) if len(differences) else None,
        "bootstrap_95_percent_ci": (
            [float(value) for value in np.percentile(means, [2.5, 97.5])]
            if len(means)
            else None
        ),
        "bootstrap_samples": 5000,
        "scenes": scene_rows,
        "classes": classes,
    }


def _save_comparison_plots(run_data: Sequence[Any], output: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        return []
    created = []
    figure, axes = plot.subplots(2, 1, figsize=(10, 8), sharex=True)
    for run, evaluations, _, _ in run_data:
        epochs = [epoch + 1 for epoch, _ in evaluations]
        axes[0].plot(
            epochs,
            [report["global"]["known_class_miou"] for _, report in evaluations],
            marker="o",
            label=run.name,
        )
        axes[1].plot(
            epochs,
            [report["mean_cross_entropy_loss"] for _, report in evaluations],
            marker="o",
            label=run.name,
        )
    axes[0].set_ylabel("Development known-class mIoU")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Development cross-entropy")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    path = output / "held_out_metrics_comparison.png"
    figure.savefig(path, dpi=160)
    plot.close(figure)
    created.append(path)
    if len(run_data) == 2:
        first = run_data[0][2][1]["global"]["per_class"]
        second = run_data[1][2][1]["global"]["per_class"]
        names = [item["name"] for item in first[1:]]
        deltas = [
            (
                float(right["iou"]) - float(left["iou"])
                if left["iou"] is not None and right["iou"] is not None
                else float("nan")
            )
            for left, right in zip(first[1:], second[1:])
        ]
        figure, axes = plot.subplots(figsize=(15, 6))
        colors = ["tab:green" if value >= 0 else "tab:red" for value in deltas]
        axes.bar(names, deltas, color=colors)
        axes.axhline(0.0, color="black", linewidth=0.8)
        axes.tick_params(axis="x", rotation=70)
        axes.set_ylabel("Second minus first IoU")
        axes.set_title("Best-checkpoint per-class change")
        figure.tight_layout()
        path = output / "per_class_iou_difference.png"
        figure.savefig(path, dpi=160)
        plot.close(figure)
        created.append(path)
    return created


def _comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines = [
        "# Training-run comparison",
        "",
        comparison["reason"],
        "",
        "## Comparability",
        "",
        f"Comparable protocol fields: **{comparison['comparability']['comparable']}**",
    ]
    lines.extend(f"- {warning}" for warning in comparison["comparability"]["warnings"])
    lines.extend(["", "## Best development results", "", _markdown_table(comparison["runs"])])
    paired = comparison.get("paired_comparison")
    if paired:
        lines.extend(
            [
                "",
                "## Paired scene result",
                "",
                f"Mean second-minus-first scene mIoU: {_format(paired['mean_paired_scene_delta'])}",
                f"Bootstrap 95% CI: {_format(paired['bootstrap_95_percent_ci'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def _comparison_html(comparison: Dict[str, Any]) -> str:
    image = (
        '<img src="held_out_metrics_comparison.png" style="max-width:100%">'
        if comparison["plots"]
        else ""
    )
    qualitative = "".join(
        (
            f"<figure><h3>{html.escape(item['run'])} — epoch {item['epoch']}</h3>"
            f"<img src='{html.escape(item['path'])}' style='max-width:100%'></figure>"
            if item["path"] is not None
            else f"<p>{html.escape(item['run'])}: no best-epoch qualitative sheet.</p>"
        )
        for item in comparison["qualitative_best_epoch"]
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Run comparison</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ccc}</style>"
        "<h1>Training-run comparison</h1>"
        f"<p>{html.escape(comparison['reason'])}</p>"
        f"<p><b>Comparable:</b> {comparison['comparability']['comparable']}</p>"
        + _html_table(comparison["runs"])
        + image
        + "<h2>Fixed development views at each selected checkpoint</h2>"
        + qualitative
        + "<p><a href='summary.json'>Machine-readable comparison</a></p>"
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return str(value)
