from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

    fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")
    ax.grid(True, which="major", color="#d8dde6", linewidth=0.9, alpha=0.75)
    ax.grid(True, which="minor", color="#eceff4", linewidth=0.6, alpha=0.8)
    ax.minorticks_on()

    # Subtle teacher-quality reference band.
    teacher_ref = by_label["Teacher Zero-Shot"]["mrr_at_10"]
    ax.axhspan(teacher_ref - 0.015, teacher_ref + 0.015, color="#eef4fb", zorder=0)
    ax.text(
        5.7,
        teacher_ref + 0.006,
        "teacher region",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4a6d92",
    )

    # Scatter all points on the main axes.
    for row in rows:
        style = family_styles[row["family"]]
        ax.scatter(
            row["query_p50_ms"],
            row["mrr_at_10"],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            edgecolors=style["edge"],
            linewidths=1.6,
            alpha=0.96,
            zorder=3,
        )

    # Main-axis labels emphasize the global story without overcrowding the student cluster.
    for label, (dx, dy) in main_label_offsets.items():
        row = by_label[label]
        ax.annotate(
            row["label"],
            (row["query_p50_ms"], row["mrr_at_10"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            color="#1f2933",
            weight="semibold" if row["family"] in {"teacher", "student_best"} else "normal",
            zorder=4,
        )

    ax.set_xlim(1.45, 7.45)
    ax.set_ylim(-0.01, 0.81)
    ax.set_xlabel("Query Encoding Latency p50 (ms)", fontsize=13)
    ax.set_ylabel("MRR@10", fontsize=13)
    ax.set_title("Latency--Quality Tradeoff Under Asymmetric Query Compression", fontsize=16, pad=14, weight="semibold")

    # Inset to resolve the low-MRR student cluster cleanly.
    inset = inset_axes(ax, width="39%", height="43%", loc="lower left", borderpad=1.8)
    inset.set_facecolor("#fffdf8")
    inset.grid(True, which="major", color="#e3e7ee", linewidth=0.7, alpha=0.8)
    for row in rows:
        style = family_styles[row["family"]]
        inset.scatter(
            row["query_p50_ms"],
            row["mrr_at_10"],
            s=max(style["size"] * 0.5, 60),
            c=style["color"],
            marker=style["marker"],
            edgecolors=style["edge"],
            linewidths=1.0,
            alpha=0.95,
            zorder=3,
        )
    inset.set_xlim(1.6, 7.35)
    inset.set_ylim(-0.002, 0.06)
    inset.tick_params(labelsize=8)
    inset.set_title("student regime", fontsize=9, pad=4)

    inset_offsets = {
        "Student Q4 CLS": (7, 7),
        "Student Q2 CLS": (7, 7),
        "Student Q4 Mean": (7, -14),
        "Student Q4 Mean + Proj": (7, 7),
        "Student Q4 Mean + Distill": (7, 7),
    }
    for label, (dx, dy) in inset_offsets.items():
        row = by_label[label]
        inset.annotate(
            label.replace("Student ", ""),
            (row["query_p50_ms"], row["mrr_at_10"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            color="#222222",
        )

    inset_progression = [
        ("Student Q4 CLS", "Student Q4 Mean", "#7b8794"),
        ("Student Q4 Mean", "Student Q4 Mean + Distill", "#2e8b57"),
        ("Student Q4 Mean", "Student Q4 Mean + Proj", "#c0392b"),
    ]
    for src_label, dst_label, color in inset_progression:
        src = by_label[src_label]
        dst = by_label[dst_label]
        inset.annotate(
            "",
            xy=(dst["query_p50_ms"], dst["mrr_at_10"]),
            xytext=(src["query_p50_ms"], src["mrr_at_10"]),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.2,
                "color": color,
                "alpha": 0.9,
                "shrinkA": 7,
                "shrinkB": 7,
                "connectionstyle": "arc3,rad=0.12",
            },
            zorder=2,
        )

    best = by_label["Student Q4 Mean"]
    inset.scatter(
        best["query_p50_ms"],
        best["mrr_at_10"],
        s=170,
        facecolors="none",
        edgecolors="#14532d",
        linewidths=1.8,
        zorder=4,
    )
    inset.text(
        2.05,
        0.056,
        "pooling recovers the first usable\nstudent signal",
        fontsize=7.5,
        color="#14532d",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#eef8f1", "edgecolor": "#9bd3ae", "linewidth": 0.8},
    )

    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#9aa5b1", lw=0.9)

    legend_handles = []
    seen = set()
    for family, style in family_styles.items():
        if style["label"] in seen:
            continue
        seen.add(style["label"])
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor=style["edge"],
                markeredgewidth=1.4,
                markersize=9,
                label=style["label"],
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d7de",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
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
    _write_pareto_plot(ROOT / "figures" / "pareto_latency_vs_mrr.png", rows)


if __name__ == "__main__":
    main()
