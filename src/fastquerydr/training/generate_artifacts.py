from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "results" / "runs"

RUN_REGISTRY = [
    {
        "label": "Teacher Zero-Shot",
        "run_dir": "teacher_bge_base_zero_shot_reference_retrieval_20260414_220138",
        "family": "teacher",
        "notes": "Frozen reference teacher",
    },
    {
        "label": "Teacher Fine-Tune",
        "run_dir": "teacher_bge_base_msmarco_finetune_conservative_20260414_221912",
        "family": "teacher_ablation",
        "notes": "Optional fine-tuned teacher ablation",
    },
    {
        "label": "Student Q4 CLS",
        "run_dir": "student_bge_query4_msmarco_20260415_195330",
        "family": "student",
        "notes": "Raw 4-layer student",
    },
    {
        "label": "Student Q2 CLS",
        "run_dir": "student_bge_query2_msmarco_20260416_170428",
        "family": "student",
        "notes": "Raw 2-layer student",
    },
    {
        "label": "Student Q4 Mean",
        "run_dir": "student_bge_query4_pool_mean_msmarco_20260416_230108",
        "family": "student_best",
        "notes": "Current best student",
    },
    {
        "label": "Student Q4 Mean + Proj",
        "run_dir": "student_bge_query4_pool_mean_proj256_msmarco_20260416_232922",
        "family": "student_ablation",
        "notes": "Projection head regression",
    },
    {
        "label": "Student Q4 Mean + Distill",
        "run_dir": "student_bge_query4_pool_mean_distill_msmarco_20260417_003541",
        "family": "student_ablation",
        "notes": "Lightweight distillation",
    },
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_rows() -> list[dict]:
    rows = []
    for entry in RUN_REGISTRY:
        run_dir = RUNS_DIR / entry["run_dir"]
        retrieval = _load_json(run_dir / "retrieval_metrics.json")
        metrics_path = run_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        latency = retrieval.get("latency", {})
        rows.append(
            {
                "label": entry["label"],
                "run_dir": entry["run_dir"],
                "family": entry["family"],
                "notes": entry["notes"],
                "mrr_at_10": retrieval["mrr_at_10"],
                "recall_at_100": retrieval["recall_at_100"],
                "query_p50_ms": latency.get("query_encode_latency_ms_p50"),
                "query_p95_ms": latency.get("query_encode_latency_ms_p95"),
                "end_to_end_p50_ms": latency.get("end_to_end_latency_ms_p50"),
                "end_to_end_p95_ms": latency.get("end_to_end_latency_ms_p95"),
                "memory_bytes": latency.get("query_memory_peak_bytes") or latency.get("query_memory_rss_delta_bytes"),
                "best_checkpoint_metric": metrics.get("best_checkpoint_metric"),
                "best_checkpoint_metric_value": metrics.get("best_checkpoint_metric_value"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_experiment_log(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Experiment Log",
        "",
        "Frozen best student reference:",
        "",
        "- Run: `student_bge_query4_pool_mean_msmarco_20260416_230108`",
        "- Rationale: clearest positive ablation result among shallow students",
        "",
        "Defensible claims:",
        "",
        "- Shrinking to 4 layers causes a large quality drop under the current setup.",
        "- Query-side pooling matters substantially for shallow query encoders.",
        "- The tested projection head hurt retrieval quality.",
        "- The tested lightweight distillation did not materially improve the tradeoff.",
        "",
        "Run Summary:",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['label']}",
                f"- Run dir: `{row['run_dir']}`",
                f"- Family: `{row['family']}`",
                f"- MRR@10: `{row['mrr_at_10']:.6f}`",
                f"- Recall@100: `{row['recall_at_100']:.6f}`",
                f"- Query p50 latency (ms): `{row['query_p50_ms']:.4f}`",
                f"- End-to-end p50 latency (ms): `{row['end_to_end_p50_ms']:.4f}`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_best_student_note(path: Path) -> None:
    lines = [
        "# Best Student Reference",
        "",
        "- Current best student run: `student_bge_query4_pool_mean_msmarco_20260416_230108`",
        "- Retrieval metrics: `MRR@10 = 0.0479215`, `Recall@100 = 0.2492314`",
        "- Reason: query-side mean pooling is the first clear positive ablation on the 4-layer student.",
        "",
        "Use this checkpoint as the student reference for Phase 7 transfer evaluation and any exact-vs-ANN comparison:",
        "",
        "- [best_model.pt](/home/cis-lab/Angad%20Singh%20Ahuja/Cloned%20Repositeries/Private/FastQueryDR/results/runs/student_bge_query4_pool_mean_msmarco_20260416_230108/best_model.pt)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pareto_plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    family_styles = {
        "teacher": {"color": "#245b9f", "marker": "o", "size": 210, "edge": "#16375f", "label": "Teacher"},
        "teacher_ablation": {"color": "#6fa8dc", "marker": "o", "size": 170, "edge": "#245b9f", "label": "Teacher Ablation"},
        "student": {"color": "#f39c12", "marker": "s", "size": 150, "edge": "#a35f00", "label": "Raw Student"},
        "student_best": {"color": "#2e8b57", "marker": "D", "size": 210, "edge": "#1d5c39", "label": "Best Student"},
        "student_ablation": {"color": "#c0392b", "marker": "^", "size": 170, "edge": "#7d251c", "label": "Student Ablation"},
    }

    main_label_offsets = {
        "Teacher Zero-Shot": (8, 10),
        "Teacher Fine-Tune": (8, -4),
        "Student Q4 CLS": (8, 8),
        "Student Q2 CLS": (8, 8),
    }

    by_label = {row["label"]: row for row in rows}

    fig = plt.figure(figsize=(4.65, 7.0), dpi=260, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.42, 0.95, 1.25])
    ax_global_top = fig.add_subplot(gs[0, 0])
    ax_global_bottom = fig.add_subplot(gs[1, 0], sharex=ax_global_top)
    ax_zoom = fig.add_subplot(gs[2, 0])
    fig.patch.set_facecolor("white")

    for ax in (ax_global_top, ax_global_bottom, ax_zoom):
        ax.set_facecolor("#fcfcfc")
        ax.grid(True, which="major", color="#d8dde6", linewidth=0.9, alpha=0.75)
        ax.grid(True, which="minor", color="#eceff4", linewidth=0.6, alpha=0.75)
        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    teacher_ref = by_label["Teacher Zero-Shot"]["mrr_at_10"]
    ax_global_top.axhspan(teacher_ref - 0.015, teacher_ref + 0.015, color="#eef4fb", zorder=0)
    ax_global_top.text(1.62, teacher_ref + 0.006, "teacher-quality band", fontsize=9.2, color="#4a6d92", ha="left")

    for row in rows:
        style = family_styles[row["family"]]
        target_ax = ax_global_top if row["mrr_at_10"] > 0.2 else ax_global_bottom
        target_ax.scatter(
            row["query_p50_ms"],
            row["mrr_at_10"],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            edgecolors=style["edge"],
            linewidths=1.5,
            alpha=0.97,
            zorder=3,
        )

    for label, (dx, dy) in main_label_offsets.items():
        row = by_label[label]
        target_ax = ax_global_top if row["mrr_at_10"] > 0.2 else ax_global_bottom
        target_ax.annotate(
            row["label"],
            (row["query_p50_ms"], row["mrr_at_10"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            color="#1f2933",
            weight="semibold",
            zorder=4,
        )

    # Show the best student in the global panel without cluttering the lower cluster.
    best = by_label["Student Q4 Mean"]
    ax_global_bottom.scatter(
        best["query_p50_ms"],
        best["mrr_at_10"],
        s=230,
        facecolors="none",
        edgecolors="#14532d",
        linewidths=2.1,
        zorder=4,
    )
    ax_global_bottom.annotate(
        "best shallow student",
        xy=(best["query_p50_ms"], best["mrr_at_10"]),
        xytext=(1.48, 0.052),
        fontsize=10,
        color="#14532d",
        ha="left",
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.5,
            "color": "#2e8b57",
            "shrinkA": 4,
            "shrinkB": 6,
            "connectionstyle": "arc3,rad=-0.22",
        },
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef8f1", "edgecolor": "#2e8b57", "linewidth": 0.9},
        zorder=5,
    )

    zoom_rect = Rectangle(
        (1.55, -0.002),
        7.35 - 1.55,
        0.06 - (-0.002),
        linewidth=1.2,
        edgecolor="#b8c2cc",
        facecolor="#f7f9fb",
        alpha=0.5,
        zorder=1,
    )
    ax_global_bottom.add_patch(zoom_rect)

    ax_global_top.set_xlim(1.45, 7.45)
    ax_global_bottom.set_xlim(1.45, 7.45)
    ax_global_top.set_ylim(0.70, 0.79)
    ax_global_bottom.set_ylim(-0.002, 0.06)
    ax_global_bottom.set_xlabel("Query encoding latency p50 (ms)", fontsize=12)
    ax_global_bottom.set_ylabel("MRR@10", fontsize=13)
    ax_global_top.text(
        0.01,
        1.02,
        "(a) Global view",
        transform=ax_global_top.transAxes,
        fontsize=12,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )
    plt.setp(ax_global_top.get_xticklabels(), visible=False)
    ax_global_top.spines["bottom"].set_visible(False)
    ax_global_bottom.spines["top"].set_visible(False)
    ax_global_top.tick_params(labeltop=False, bottom=False)
    ax_global_bottom.tick_params(top=False)

    # Broken-axis marks.
    d = 0.012
    kwargs = dict(transform=ax_global_top.transAxes, color="#44546a", clip_on=False, linewidth=1.0)
    ax_global_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_global_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_global_bottom.transAxes)
    ax_global_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_global_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Student regime panel.
    student_labels = [
        "Student Q4 CLS",
        "Student Q2 CLS",
        "Student Q4 Mean",
        "Student Q4 Mean + Proj",
        "Student Q4 Mean + Distill",
    ]
    for label in student_labels:
        row = by_label[label]
        style = family_styles[row["family"]]
        ax_zoom.scatter(
            row["query_p50_ms"],
            row["mrr_at_10"],
            s=max(style["size"] * 0.8, 100),
            c=style["color"],
            marker=style["marker"],
            edgecolors=style["edge"],
            linewidths=1.3,
            alpha=0.97,
            zorder=3,
        )

    zoom_offsets = {
        "Student Q4 CLS": (8, 7),
        "Student Q2 CLS": (8, 7),
        "Student Q4 Mean": (8, -14),
        "Student Q4 Mean + Proj": (8, 7),
        "Student Q4 Mean + Distill": (8, 7),
    }
    for label, (dx, dy) in zoom_offsets.items():
        row = by_label[label]
        ax_zoom.annotate(
            label.replace("Student ", ""),
            (row["query_p50_ms"], row["mrr_at_10"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            color="#222222",
        )

    student_frontier = ["Student Q4 Mean + Proj", "Student Q4 Mean", "Student Q4 Mean + Distill"]
    ax_zoom.plot(
        [by_label[l]["query_p50_ms"] for l in student_frontier],
        [by_label[l]["mrr_at_10"] for l in student_frontier],
        color="#7b8794",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )

    transitions = [
        ("Student Q4 CLS", "Student Q4 Mean", "#7b8794"),
        ("Student Q4 Mean", "Student Q4 Mean + Distill", "#2e8b57"),
        ("Student Q4 Mean", "Student Q4 Mean + Proj", "#c0392b"),
    ]
    for src_label, dst_label, color in transitions:
        src = by_label[src_label]
        dst = by_label[dst_label]
        ax_zoom.annotate(
            "",
            xy=(dst["query_p50_ms"], dst["mrr_at_10"]),
            xytext=(src["query_p50_ms"], src["mrr_at_10"]),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.3,
                "color": color,
                "alpha": 0.9,
                "shrinkA": 8,
                "shrinkB": 8,
                "connectionstyle": "arc3,rad=0.14",
            },
            zorder=2,
        )

    ax_zoom.text(
        1.92,
        0.057,
        "pooling produces the first\nusable student recovery",
        fontsize=8.7,
        color="#14532d",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "#eef8f1", "edgecolor": "#9bd3ae", "linewidth": 0.8},
    )
    ax_zoom.set_xlim(1.55, 7.35)
    ax_zoom.set_ylim(-0.002, 0.06)
    ax_zoom.set_xlabel("Query encoding latency p50 (ms)", fontsize=12)
    ax_zoom.text(
        0.01,
        1.02,
        "(b) Student regime zoom",
        transform=ax_zoom.transAxes,
        fontsize=12,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )

    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_ann_plot(path: Path, ann_table_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    ann = pd.read_csv(ann_table_path)
    style_map = {
        "flat": {"color": "#245b9f", "marker": "o", "edge": "#16375f"},
        "hnsw": {"color": "#3b82f6", "marker": "s", "edge": "#1d4ed8"},
        "ivf": {"color": "#d97706", "marker": "^", "edge": "#92400e"},
    }

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=260, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")
    ax.grid(True, which="major", color="#d8dde6", linewidth=0.85, alpha=0.8)
    ax.grid(True, which="minor", color="#eceff4", linewidth=0.55, alpha=0.75)
    ax.minorticks_on()
    ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    flat_mrr = float(ann.loc[ann["index_type"] == "flat", "mrr_at_10"].iloc[0])
    ax.axhspan(flat_mrr * 0.92, flat_mrr * 1.01, color="#eef4fb", alpha=0.8, zorder=0)
    ax.axhline(flat_mrr, color="#7b8794", linestyle="--", linewidth=1.15, alpha=0.85)
    ax.text(0.18, flat_mrr + 0.00035, "near-exact quality band", fontsize=8.8, color="#6b7280", ha="left")

    for _, row in ann.iterrows():
        idx = row["index_type"].lower()
        style = style_map[idx]
        ax.scatter(
            row["mean_search_ms"],
            row["mrr_at_10"],
            s=120,
            c=style["color"],
            marker=style["marker"],
            edgecolors=style["edge"],
            linewidths=1.4,
            zorder=3,
        )

    label_offsets = {
        "flat": (6, 4),
        "hnsw": (6, 8),
        "ivf": (6, 3),
    }
    for _, row in ann.iterrows():
        idx = row["index_type"].lower()
        dx, dy = label_offsets[idx]
        ax.annotate(
            row["index_type"].upper() if idx != "flat" else "Flat",
            (row["mean_search_ms"], row["mrr_at_10"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            color=style_map[idx]["edge"],
            weight="semibold",
        )

    ivf = ann.loc[ann["index_type"] == "ivf"].iloc[0]
    hnsw = ann.loc[ann["index_type"] == "hnsw"].iloc[0]
    ax.annotate(
        "IVF: $\\sim 3.1\\times$ faster,\nsmall quality drop",
        xy=(ivf["mean_search_ms"], ivf["mrr_at_10"]),
        xytext=(0.16, 0.0463),
        fontsize=8.8,
        color="#92400e",
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.1,
            "color": "#d97706",
            "connectionstyle": "arc3,rad=-0.15",
        },
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff7ed", "edgecolor": "#f59e0b", "linewidth": 0.8},
    )
    ax.annotate(
        "HNSW: much faster,\nquality too low",
        xy=(hnsw["mean_search_ms"], hnsw["mrr_at_10"]),
        xytext=(0.055, 0.0344),
        fontsize=8.5,
        color="#1d4ed8",
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.0,
            "color": "#3b82f6",
            "connectionstyle": "arc3,rad=0.18",
        },
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#eff6ff", "edgecolor": "#93c5fd", "linewidth": 0.8},
    )

    ax.set_xlim(0.03, 5.0)
    ax.set_ylim(0.028, 0.0505)
    ax.set_xlabel("Mean search latency (ms, log scale)", fontsize=11)
    ax.set_ylabel("MRR@10", fontsize=11)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _collect_rows()
    _write_csv(
        ROOT / "results" / "main_table.csv",
        rows,
        ["label", "family", "mrr_at_10", "recall_at_100", "notes", "run_dir"],
    )
    _write_csv(
        ROOT / "results" / "latency_table.csv",
        rows,
        ["label", "family", "query_p50_ms", "query_p95_ms", "end_to_end_p50_ms", "end_to_end_p95_ms", "memory_bytes", "run_dir"],
    )
    _write_experiment_log(ROOT / "notes" / "experiment_log.md", rows)
    _write_best_student_note(ROOT / "notes" / "best_student_reference.md")
    figure_dir = ROOT / "paper" / "figures"
    _write_pareto_plot(figure_dir / "pareto_latency_vs_mrr.png", rows)
    ann_table_path = ROOT / "results" / "ann_table.csv"
    if ann_table_path.exists():
        _write_ann_plot(figure_dir / "ann_search_latency_vs_mrr.png", ann_table_path)


if __name__ == "__main__":
    main()
