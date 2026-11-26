from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# Every number in docs/ric/ is generated here from experiments/ric_clover/results/,
# so the write-up can be rebuilt after any rerun: python -m experiments.ric_clover.
# make_report_assets (from the repository root).

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "docs", "ric")

# Validated with the dataviz skill's validate_palette.js (light, all checks pass).
RIC_BLUE = "#2a78d6"
BASE_ORANGE = "#eb6834"
REALIZED_GRAY = "#8a8984"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

CLASS_LABELS = ["different region", "same region,\nother gender", "siblings\n(FT vs PT)"]
SYN_LABELS = ["different region", "same region,\nother gender", "siblings\n(FT vs PT)"]


def records(name: str) -> list[dict]:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def index(rows: list[dict]) -> dict:
    out = defaultdict(dict)
    for r in rows:
        out[(r["dataset"], r["variant"])][r["seed"]] = r
    return out


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired(idx: dict, dataset: str, variant: str) -> dict | None:
    base, cand = idx.get((dataset, "base"), {}), idx.get((dataset, variant), {})
    seeds = sorted(set(base) & set(cand))
    if len(seeds) < 3:
        return None
    b = np.array([base[s]["scrps"]["Overall"] for s in seeds])
    c = np.array([cand[s]["scrps"]["Overall"] for s in seeds])
    rel = c / b - 1
    return {
        "n": len(seeds),
        "base": b.mean(),
        "var": c.mean(),
        "mean": rel.mean() * 100,
        "median": np.median(rel) * 100,
        "wins": int((rel < 0).sum()),
        "p": float(stats.wilcoxon(c, b).pvalue),
    }


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main_table() -> str:
    rows = [
        ("Labour", "last 12 months", "labour_feat_lasth.jsonl", "labour"),
        ("Labour", "4 rolling origins", "labour_feat_roll4.jsonl", "labour"),
        ("Labour, no features", "last 12 months", "labour_lasth.jsonl", "labour"),
        ("Labour, no features", "4 rolling origins", "labour_roll4.jsonl", "labour"),
        ("Prison", "last 8 quarters", "others_lasth.jsonl", "prison"),
        ("Tourism-S", "last 4 quarters", "others_lasth.jsonl", "tourism_small"),
        ("Wiki2", "last 7 days", "others_lasth.jsonl", "wiki2"),
        ("Traffic", "last day", "others_lasth.jsonl", "traffic"),
        ("Traffic", "7 rolling days", "traffic_roll7.jsonl", "traffic"),
        ("Synthetic", "4 rolling origins", "synthetic_ric.jsonl", "synthetic_ric"),
    ]
    variants = [("ric_tree", "RIC"), ("ric_tree_att1", r"RIC, $\rho{=}0.8$")]
    cells = []
    for label, protocol, path, dataset in rows:
        idx = index(records(path))
        for v, _ in variants:
            cells.append(((label, protocol, v), paired(idx, dataset, v)))
    adjusted = holm([c["p"] for _, c in cells])
    lines = [
        r"\begin{tabular}{llr" + "rrr" * len(variants) + "}",
        r"\toprule",
        r" & & & "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in variants)
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(variants))
        ),
        r"Dataset & Test protocol & CLOVER & $\Delta$ & wins & $p_\text{Holm}$"
        + r" & $\Delta$ & wins & $p_\text{Holm}$ \\",
        r"\midrule",
    ]
    k = 0
    for label, protocol, _, _ in rows:
        parts = [label, protocol]
        first = cells[k][1]
        parts.append(f"{first['base']:.4f}")
        for _ in variants:
            c, p_adj = cells[k][1], adjusted[k]
            delta = f"{c['mean']:+.1f}\\%"
            if p_adj < 0.05:
                delta = rf"\textbf{{{delta}}}"
            parts += [delta, f"{c['wins']}/{c['n']}", fmt_p(p_adj)]
            k += 1
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ablation_table() -> str:
    cols = [
        ("labour_lasth.jsonl", "labour", "Labour, last 12 m"),
        ("labour_roll4.jsonl", "labour", "Labour, rolling"),
        ("synthetic_ric.jsonl", "synthetic_ric", "Synthetic"),
    ]
    rows = [
        ("ric_tree", r"RIC: differences, tree target, analytic shrinkage"),
        ("ric_lw", r"\quad identity target instead of tree"),
        ("ric", r"\quad fixed shrinkage $\lambda{=}0.5$ toward $I$"),
        ("ric_tree_level", r"\quad levels instead of differences"),
        ("ric_level", r"\quad levels, identity target, $\lambda{=}0.5$"),
        ("independent", r"\quad no cross-series dependence at all"),
        ("hybrid", r"\quad gated mix with the network's own loadings"),
        ("ric_tree_multi", r"\quad $\eta$-step changes for horizon $\eta$"),
        ("ric_tree_m60", r"\quad 60-row estimation history"),
        ("ric_tree_m120", r"\quad 120-row estimation history"),
        ("ric_tree_all", r"\quad full history"),
        ("ric_tree_att1", r"\quad attenuation $\rho{=}0.8$"),
        ("ric_tree_rho65", r"\quad attenuation $\rho{=}0.65$"),
        ("ric_tree_rho50", r"\quad attenuation $\rho{=}0.5$"),
        ("ric_tree_att", r"\quad attenuation, learned $\rho_\eta$"),
    ]
    idx = {path: index(records(path)) for path, _, _ in cols}
    lines = [
        r"\begin{tabular}{l" + "rr" * len(cols) + "}",
        r"\toprule",
        " & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for _, _, t in cols)
        + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
        "Variant" + r" & $\Delta$ & $n$" * len(cols) + r" \\",
        r"\midrule",
    ]
    for variant, label in rows:
        parts = [label]
        for path, dataset, _ in cols:
            c = paired(idx[path], dataset, variant)
            parts += ["--", ""] if c is None else [f"{c['mean']:+.1f}\\%", str(c["n"])]
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def level_table() -> str:
    cols = [
        ("labour_feat_lasth.jsonl", "last 12 months"),
        ("labour_feat_roll4.jsonl", "rolling"),
    ]
    levels = ["Total", "Geography", "Gender/Geography", "Status/Gender/Geography"]
    names = ["Total", "Geography", "Gender $\\times$ Geography", "Bottom (status)"]
    lines = [
        r"\begin{tabular}{l" + "rrrr" * len(cols) + "}",
        r"\toprule",
        " & "
        + " & ".join(rf"\multicolumn{{4}}{{c}}{{Labour, {t}}}" for _, t in cols)
        + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(cols))),
        r" & \multicolumn{2}{c}{sCRPS} & \multicolumn{2}{c}{spread/RMSE}" * len(cols)
        + r" \\",
        r"Level" + r" & CLOVER & RIC & CLOVER & RIC" * len(cols) + r" \\",
        r"\midrule",
    ]
    data = {}
    for path, _ in cols:
        idx = index(records(path))
        b, r = idx[("labour", "base")], idx[("labour", "ric_tree_att1")]
        seeds = sorted(set(b) & set(r))
        data[path] = (
            {lv: np.mean([b[s]["scrps"][lv] for s in seeds]) for lv in levels},
            {lv: np.mean([r[s]["scrps"][lv] for s in seeds]) for lv in levels},
            {
                lv: np.mean([b[s]["calibration"][lv]["spread_rmse"] for s in seeds])
                for lv in levels
            },
            {
                lv: np.mean([r[s]["calibration"][lv]["spread_rmse"] for s in seeds])
                for lv in levels
            },
        )
    for lv, name in zip(levels, names, strict=True):
        parts = [name]
        for path, _ in cols:
            sb, sr, cb, cr = data[path]
            parts += [f"{sb[lv]:.4f}", f"{sr[lv]:.4f}", f"{cb[lv]:.2f}", f"{cr[lv]:.2f}"]
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def correlation_figure() -> None:
    panels = [
        ("corr_labour.json", "Labour (test windows)"),
        ("corr_synthetic.json", "Synthetic"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    width = 0.26
    for ax, (path, title) in zip(axes, panels, strict=True):
        with open(os.path.join(RESULTS, path), encoding="utf-8") as handle:
            d = json.load(handle)
        classes = ["0", "1", "2"]
        realized = [d["base"]["realized"][c] for c in classes]
        base_h1 = [d["base"]["predicted"]["1"][c] for c in classes]
        base_h12 = [d["base"]["predicted"]["12"][c] for c in classes]
        ric = [d["ric_tree"]["predicted"]["1"][c] for c in classes]
        x = np.arange(3)
        series = [
            (x - width - 0.02, realized, REALIZED_GRAY, "realized innovations"),
            (x, base_h1, BASE_ORANGE, "CLOVER (h=1)"),
            (x + width + 0.02, ric, RIC_BLUE, "RIC"),
        ]
        for pos, vals, color, label in series:
            ax.bar(
                pos,
                vals,
                width,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=1.0,
            )
        # CLOVER's correlation keeps growing with the horizon: mark h=12 as a tick.
        ax.scatter(
            x,
            base_h12,
            marker="_",
            s=180,
            color=BASE_ORANGE,
            linewidths=2,
            label="CLOVER (h=12)",
            zorder=3,
        )
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_LABELS, fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        style(ax)
        for pos, vals in ((x - width - 0.02, realized), (x + width + 0.02, ric)):
            i = 2
            ax.annotate(
                f"{vals[i]:+.2f}",
                (pos[i], vals[i]),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                fontsize=7,
                color=TEXT_2,
            )
        ax.annotate(
            f"{base_h1[2]:+.2f}",
            (x[2], base_h1[2]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
            color=TEXT_2,
        )
    axes[0].set_ylabel("mean pairwise correlation", fontsize=8, color=TEXT)
    axes[0].set_ylim(-0.8, 0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    wanted = ("realized innovations", "CLOVER (h=1)", "CLOVER (h=12)", "RIC")
    order = [labels.index(name) for name in wanted]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_correlation.{ext}"), dpi=200)
    plt.close(fig)


def calibration_figure() -> None:
    panels = [
        (
            "labour_roll4.jsonl",
            "labour",
            "Labour (rolling origins)",
            ["Total", "Geography", "Gender/Geography", "Status/Gender/Geography"],
        ),
        (
            "synthetic_ric.jsonl",
            "synthetic_ric",
            "Synthetic",
            ["Total", "Region", "Region/Gender", "Bottom"],
        ),
    ]
    names = ["Total", "Level 2", "Level 3", "Bottom"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (path, dataset, title, levels) in zip(axes, panels, strict=True):
        idx = index(records(path))
        y = np.arange(len(levels))[::-1]
        for variant, color, label, dy in (
            ("base", BASE_ORANGE, "CLOVER", 0.12),
            ("ric_tree", RIC_BLUE, "RIC", -0.12),
        ):
            runs = idx[(dataset, variant)]
            vals = [
                np.mean([r["calibration"][lv]["spread_rmse"] for r in runs.values()])
                for lv in levels
            ]
            ax.scatter(
                vals,
                y + dy,
                s=36,
                color=color,
                label=label,
                zorder=3,
                edgecolors="white",
                linewidths=1.0,
            )
            for v, yy in zip(vals, y + dy, strict=True):
                ax.annotate(
                    f"{v:.2f}",
                    (v, yy),
                    textcoords="offset points",
                    xytext=(6, -3),
                    fontsize=7,
                    color=TEXT_2,
                )
        ax.axvline(1.0, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_yticks(np.arange(len(levels))[::-1])
        ax.set_yticklabels(names, fontsize=8, color=TEXT)
        ax.set_xlim(0.6, 2.4)
        ax.set_xlabel(
            "predictive spread / RMSE  (1 = calibrated)", fontsize=8, color=TEXT
        )
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_calibration.{ext}"), dpi=200)
    plt.close(fig)


def joint_table() -> str:
    """Energy and variogram score of the joint bottom distribution (`bench --joint`)."""
    rows = [
        ("Labour", "last 12 months", "joint_labour_feat_lasth.jsonl", "labour"),
        ("Labour", "4 rolling origins", "joint_labour_feat_roll4.jsonl", "labour"),
        ("Prison", "last 8 quarters", "joint_others_lasth.jsonl", "prison"),
        ("Tourism-S", "last 4 quarters", "joint_others_lasth.jsonl", "tourism_small"),
        ("Wiki2", "last 7 days", "joint_others_lasth.jsonl", "wiki2"),
        ("Traffic", "7 rolling days", "joint_traffic_roll7.jsonl", "traffic"),
        ("Synthetic", "4 rolling origins", "joint_synthetic.jsonl", "synthetic_ric"),
    ]
    metrics = [("energy", "Energy score"), ("variogram", "Variogram score")]
    variants = [("ric_tree", "RIC"), ("ric_tree_att1", r"$\rho{=}0.8$")]

    def by_metric(path: str, metric: str) -> dict:
        out = defaultdict(dict)
        for r in records(path):
            out[(r["dataset"], r["variant"])][r["seed"]] = {
                "scrps": {"Overall": r["joint"][metric]}
            }
        return out

    cells = []
    for _, _, path, dataset in rows:
        for metric, _ in metrics:
            idx = by_metric(path, metric)
            for v, _ in variants:
                cells.append(paired(idx, dataset, v))
    adjusted = holm([c["p"] for c in cells])
    n_cols = len(metrics) * len(variants)
    lines = [
        r"\begin{tabular}{ll" + "rr" * n_cols + "}",
        r"\toprule",
        r" & & "
        + " & ".join(
            rf"\multicolumn{{{2 * len(variants)}}}{{c}}{{{name}}}" for _, name in metrics
        )
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{3 + 2 * len(variants) * i}-"
            rf"{2 + 2 * len(variants) * (i + 1)}}}"
            for i in range(len(metrics))
        ),
        r" & & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{name}}}" for _ in metrics for _, name in variants
        )
        + r" \\",
        r"Dataset & Test protocol" + r" & $\Delta$ & wins" * n_cols + r" \\",
        r"\midrule",
    ]
    k = 0
    for label, protocol, _, _ in rows:
        parts = [label, protocol]
        for _ in range(n_cols):
            c, p_adj = cells[k], adjusted[k]
            delta = f"{c['mean']:+.1f}\\%"
            if p_adj < 0.05:
                delta = rf"\textbf{{{delta}}}"
            parts += [delta, f"{c['wins']}/{c['n']}"]
            k += 1
        lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def gate_table() -> str:
    """Pre-test backtest gate: what each selection score picks, and the real-test result.

    Pool: CLOVER, RIC, RIC with rho 0.8. Labour's test is the rolling protocol.
    """
    import sys

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from experiments.ric_clover.gate_eval import load as load_scores

    gate = load_scores([os.path.join(RESULTS, "gate_joint.jsonl")])
    test = load_scores(
        [
            os.path.join(RESULTS, f)
            for f in (
                "joint_labour_feat_roll4.jsonl",
                "joint_others_lasth.jsonl",
                "joint_traffic_roll7.jsonl",
            )
        ]
    )
    names = {
        "labour": "Labour",
        "prison": "Prison",
        "tourism_small": "Tourism-S",
        "wiki2": "Wiki2",
        "traffic": "Traffic",
    }
    short = {"base": "CLOVER", "ric_tree": "RIC", "ric_tree_att1": r"RIC $\rho{=}0.8$"}
    pool = ["base", "ric_tree", "ric_tree_att1"]

    def pick(dataset: str, metric: str) -> str:
        scores = {
            v: np.mean([s[metric] for s in gate[(dataset, v)].values()])
            for v in pool
            if gate.get((dataset, v))
        }
        return min(scores, key=scores.get)

    def change(dataset: str, variant: str, metric: str) -> str:
        base, chosen = test[(dataset, "base")], test[(dataset, variant)]
        seeds = [s for s in base if s in chosen]
        rel = np.array([chosen[s][metric] / base[s][metric] - 1 for s in seeds])
        return f"{rel.mean() * 100:+.1f}\\% ({(rel < 0).sum()}/{len(seeds)})"

    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r" & \multicolumn{2}{c}{sCRPS-selected} & \multicolumn{3}{c}{energy-selected} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"Dataset & pick & test energy & pick & test sCRPS & test energy \\",
        r"\midrule",
    ]
    for d, label in names.items():
        p_s, p_e = pick(d, "scrps"), pick(d, "energy")
        lines.append(
            " & ".join(
                [
                    label,
                    short[p_s],
                    change(d, p_s, "energy"),
                    short[p_e],
                    change(d, p_e, "scrps"),
                    change(d, p_e, "energy"),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    for name, build in (
        ("main", main_table),
        ("ablation", ablation_table),
        ("levels", level_table),
        ("joint", joint_table),
        ("gate", gate_table),
    ):
        with open(os.path.join(OUT, "tables", f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(build() + "\n")
    correlation_figure()
    calibration_figure()
    print(f"wrote tables and figures to {OUT}")


if __name__ == "__main__":
    main()
