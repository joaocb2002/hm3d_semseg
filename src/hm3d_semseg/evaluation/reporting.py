"""Static human-facing report for one explicit evaluation command."""

# HTML/CSS template lines remain intact for readability.
# ruff: noqa: E501

from __future__ import annotations

import csv
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hm3d_semseg.utils.hashing import atomic_write_json, atomic_write_text, sha256_file


def generate_evaluation_report(output: Path) -> Dict[str, Any]:
    """Build tables and HTML from an existing explicit evaluation summary."""
    output = output.resolve()
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Evaluation summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_root = output / "report"
    tables = report_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_class = summary["global"]["per_class"]
    objectnav = [
        {"goal": goal, **values}
        for goal, values in summary["global"]["objectnav_six"].items()
    ]
    scenes = [
        {"scene_id": scene, "known_class_miou": value}
        for scene, value in sorted(
            summary["scene_macro"]["per_scene_known_class_miou"].items()
        )
    ]
    confusions = _top_confusions(summary)
    table_paths = [
        _write_csv(tables / "per_class.csv", per_class),
        _write_csv(tables / "objectnav_six.csv", objectnav),
        _write_csv(tables / "per_scene.csv", scenes),
        _write_csv(tables / "top_confusions.csv", confusions),
    ]
    headline = {
        "mean_cross_entropy_loss": summary.get("mean_cross_entropy_loss"),
        "known_class_miou": summary["global"].get("known_class_miou"),
        "miou_41": summary["global"].get("miou_41"),
        "objectnav_six_miou": summary["global"].get("objectnav_six_miou"),
        "overall_pixel_accuracy": summary["global"].get("overall_pixel_accuracy"),
        "frequency_weighted_iou": summary["global"].get(
            "frequency_weighted_iou"
        ),
        "mean_class_recall": summary["global"].get("mean_class_recall"),
        "unknown_iou": summary["global"].get("unknown", {}).get("iou"),
        "unknown_prevalence": summary["global"].get("unknown", {}).get(
            "prevalence"
        ),
        "scene_macro_mean_miou": summary["scene_macro"].get("mean"),
        "scene_macro_median_miou": summary["scene_macro"].get("median"),
        "scene_bootstrap_95_percent_ci": summary["scene_macro"].get(
            "bootstrap_95_percent_ci"
        ),
        "nll": summary["probability_quality"].get("nll"),
        "multiclass_brier": summary["probability_quality"].get(
            "multiclass_brier"
        ),
        "ece": summary["probability_quality"].get("ece"),
    }
    report_summary = {
        "schema_version": "1.0",
        "evaluation": str(output),
        "checkpoint": summary.get("checkpoint"),
        "dataset": summary.get("dataset"),
        "samples": summary.get("evaluation_samples"),
        "scenes": summary.get("evaluation_scenes"),
        "temperature": summary.get("temperature"),
        "headline": headline,
        "top_confusions": confusions,
        "qualitative": summary.get("qualitative"),
    }
    atomic_write_json(report_root / "summary.json", report_summary)
    atomic_write_text(report_root / "summary.md", _markdown(report_summary))
    atomic_write_text(
        report_root / "index.html",
        _html(report_summary, per_class, objectnav, scenes),
    )
    generated = [
        report_root / "summary.json",
        report_root / "summary.md",
        report_root / "index.html",
        *table_paths,
    ]
    atomic_write_json(
        report_root / "report_manifest.json",
        {
            "schema_version": "1.0",
            "source": str(summary_path),
            "source_sha256": sha256_file(summary_path),
            "generated": [
                {
                    "path": str(path.relative_to(report_root)),
                    "sha256": sha256_file(path),
                }
                for path in generated
            ],
        },
    )
    return {
        "report": str(report_root / "index.html"),
        "summary": str(report_root / "summary.md"),
        "tables": len(table_paths),
    }


def _top_confusions(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    matrix = summary["global"]["row_normalized_confusion_matrix"]
    classes = summary["global"]["per_class"]
    rows = []
    for true_id, row in enumerate(matrix):
        for predicted_id, fraction in enumerate(row):
            if true_id == predicted_id or float(fraction) <= 0.0:
                continue
            rows.append(
                {
                    "true_id": true_id,
                    "true_class": classes[true_id]["name"],
                    "predicted_id": predicted_id,
                    "predicted_class": classes[predicted_id]["name"],
                    "fraction_of_true_class": float(fraction),
                }
            )
    return sorted(
        rows, key=lambda item: float(item["fraction_of_true_class"]), reverse=True
    )[:30]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    columns = list(rows[0]) if rows else []
    buffer = io.StringIO(newline="")
    if columns:
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return path


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Scope: {report['samples']} samples across {report['scenes']} scenes",
        f"- Temperature: {_format(report['temperature'])}",
        "",
        "## Headline metrics",
        "",
    ]
    lines.extend(
        f"- {name.replace('_', ' ')}: {_format(value)}"
        for name, value in report["headline"].items()
    )
    lines.extend(
        [
            "",
            "Open [the HTML report](index.html) for plots, sortable tables, and "
            "qualitative views.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(
    report: Dict[str, Any],
    per_class: Sequence[Dict[str, Any]],
    objectnav: Sequence[Dict[str, Any]],
    scenes: Sequence[Dict[str, Any]],
) -> str:
    plot_cards = "".join(
        f"<figure><img src='../plots/{html.escape(path.name)}'><figcaption>"
        f"{html.escape(path.stem.replace('_', ' '))}</figcaption></figure>"
        for path in sorted((Path(report["evaluation"]) / "plots").glob("*.png"))
    )
    qualitative = report.get("qualitative")
    qualitative_html = "<p>Not recorded.</p>"
    if qualitative is not None:
        contact = qualitative.get("contact_sheet")
        if contact:
            qualitative_html = (
                "<img class='contact' src='../qualitative/"
                + html.escape(str(contact))
                + "'>"
            )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Evaluation report</title>
<style>body{font:15px/1.5 system-ui;background:#f4f6f8;color:#18202a;margin:0}
main{max-width:1280px;margin:auto;padding:28px}.card,figure{background:white;border:1px solid #d9dfe7;border-radius:10px;padding:15px;margin:15px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:15px}img{max-width:100%}.contact{width:100%}.table-wrap{overflow:auto;max-height:600px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#e8edf3}th:first-child,td:first-child{text-align:left}</style></head><body><main>""" + (
        f"<h1>Evaluation report</h1><section class='card'><p><b>Checkpoint:</b> {html.escape(str(report['checkpoint']))}<br><b>Dataset:</b> {html.escape(str(report['dataset']))}<br><b>Scope:</b> {report['samples']} samples / {report['scenes']} scenes<br><b>Temperature:</b> {_format(report['temperature'])}</p>{_table([report['headline']])}</section>"
        f"<h2>Qualitative fixed set</h2><section class='card'>{qualitative_html}</section>"
        f"<h2>Plots</h2><section class='grid'>{plot_cards}</section>"
        f"<section class='card'><h2>ObjectNav six</h2><div class='table-wrap'>{_table(objectnav)}</div></section>"
        f"<section class='card'><h2>Per-class metrics</h2><div class='table-wrap'>{_table(per_class)}</div></section>"
        f"<section class='card'><h2>Per-scene mIoU</h2><div class='table-wrap'>{_table(scenes)}</div></section>"
        f"<section class='card'><h2>Largest confusions</h2><div class='table-wrap'>{_table(report['top_confusions'])}</div></section>"
        "<p><a href='../summary.json'>Authoritative evaluation JSON</a></p></main></body></html>"
    )


def _table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(name.replace('_', ' '))}</th>" for name in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(row.get(column)))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(item) for item in value) + "]"
    return str(value)
