"""Present completed frozen statistics; no fitting, scoring or model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.probability_phase4_data import sha256


def choose(frame, **values):
    keep = np.ones(len(frame), dtype=bool)
    for key, value in values.items():
        keep &= frame[key].eq(value).to_numpy()
    return frame.loc[keep]


def number(value, digits=4):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]]) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/probability_phase4"))
    a = p.parse_args(argv)
    out = a.root / "evaluation"
    state = json.loads((out / "execution_state.json").read_text())
    if state["status"] != "completed":
        raise ValueError("Only a completed formal evaluation may be reported")
    filenames = ["annual_scores", "aggregate_scores", "comparisons", "annual_comparisons",
                 "bootstrap_intervals", "reliability", "aggregate_alert_metrics"]
    data = {name: pd.read_csv(out / (name + ".csv")) for name in filenames}
    gates = json.loads((out / "application_gate.json").read_text())
    calibrators = json.loads((a.root / "calibration/calibrators.json").read_text())
    figures = Path("figures/probability_phase4")
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "ehhr-phase4-v1",
                         "axes.spines.top": False, "axes.spines.right": False})
    saved = []
    def save(fig, stem):
        fig.tight_layout()
        for suffix in ("svg", "png"):
            path = figures / f"{stem}.{suffix}"
            fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Date": None} if suffix == "svg" else None)
            saved.append(path)
        plt.close(fig)
    compare = data["comparisons"]
    ci = data["bootstrap_intervals"]
    fixed = dict(candidate="lightgbm_F4", reference="logistic_F4",
                 candidate_probability="raw", reference_probability="raw", weighting="equal_grid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    comparison_rows = []
    for ax, scope in zip(axes, ("temporal", "strict")):
        labels = []
        for position, (lag, risk) in enumerate([(1, "all"), (1, "onset"), (2, "all"), (2, "onset")]):
            point = choose(compare, scope=scope, lag=lag, risk_set=risk, **fixed).iloc[0]
            interval = choose(ci, scope=scope, lag=lag, risk_set=risk, block_days=14, **fixed).iloc[0]
            delta, lo, hi = point.delta_brier, interval.delta_brier_lower, interval.delta_brier_upper
            if np.isfinite([lo, hi]).all():
                ax.plot([lo, hi], [position, position], color="#627d98", linewidth=2)
            ax.plot(delta, position, "o", color="#176184")
            labels.append(f"D-{lag} {risk}")
            comparison_rows.append([scope, f"D−{lag}", risk, number(point.candidate_brier),
                                    number(point.reference_brier), number(delta),
                                    f"[{number(lo)}, {number(hi)}]", f"{int(point.years_improving)}/5"])
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.5, -.5)
        ax.set_title("Temporal: all land" if scope == "temporal" else "Strict: H holdout")
        ax.set_xlabel("Raw Brier difference: LightGBM F4 - LR F4")
        ax.grid(axis="x", alpha=.15)
    save(fig, "fixed_f4_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    annual = data["annual_comparisons"]
    moisture_rows = []
    styles = [("lightgbm_F1", "lightgbm_F0", "LGB moisture", "#176184", "-"),
              ("lightgbm_F1P", "lightgbm_F0P", "LGB conditional", "#176184", "--"),
              ("logistic_F1", "logistic_F0", "LR moisture", "#c8652c", "-"),
              ("logistic_F1P", "logistic_F0P", "LR conditional", "#c8652c", "--")]
    for ax, scope in zip(axes, ("temporal", "strict")):
        for candidate, reference, label, color, linestyle in styles:
            keys = dict(scope=scope, lag=2, risk_set="onset", candidate=candidate, reference=reference,
                        candidate_probability="raw", reference_probability="raw", weighting="equal_grid")
            yearly = choose(annual, **keys).sort_values("year")
            point = choose(compare, **keys).iloc[0]
            interval = choose(ci, block_days=14, **keys).iloc[0]
            ax.plot(yearly.year, yearly.delta_brier, marker="o", markersize=4, label=label, color=color, linestyle=linestyle)
            moisture_rows.append([scope, label, number(point.delta_brier),
                f"[{number(interval.delta_brier_lower)}, {number(interval.delta_brier_upper)}]",
                f"{int(point.years_improving)}/5"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(2020, 2025))
        ax.set_title(scope + ": D-2 onset risk set")
        ax.set_ylabel("Raw Brier difference (added moisture)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    save(fig, "lag2_onset_moisture_years")

    reliability = data["reliability"]
    for scope in ("temporal", "strict"):
        for lag in (1, 2):
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True, sharey=True)
            for ax, year in zip(axes, range(2020, 2025)):
                ax.plot([0, 1], [0, 1], ":", color="grey", linewidth=1)
                for model, color in [("logistic_F4", "#c8652c"), ("lightgbm_F4", "#176184")]:
                    for kind, style in [("raw", "--"), ("calibrated", "-")]:
                        rows = choose(reliability, scope=scope, lag=lag, model_id=model,
                                      year=year, probability=kind, risk_set="all").sort_values("bin")
                        # Preserve empty bins as NaN gaps, without interpolation.
                        ax.plot(rows.mean_probability, rows.event_fraction, style, color=color,
                                marker="o", markersize=3, label=model.replace("logistic", "LR").replace("lightgbm", "LGB") + " " + kind)
                ax.set_title(str(year))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Mean predicted probability")
                ax.grid(alpha=.15)
            axes[0].set_ylabel("Observed event fraction")
            axes[-1].legend(fontsize=7, loc="lower right")
            fig.suptitle(f"{scope} D-{lag}: annual F4 reliability (fixed 10 bins)", y=1.03)
            save(fig, f"reliability_{scope}_lag{lag}")

    lines = ["# 第四阶段冻结校准与正式评价结果", "",
             "本报告仅整理已完成的冻结评价；不据测试成绩重选模型、特征或阈值。ERA5结果属于概率历史回报。", "",
             "## 执行完成与数据分离", "",
             f"正式评价覆盖 {state['model_instances']} 个固定配置、{state['model_year_bundles']} 份模型—年预测、{state['common_indices']} 份scope—年共同索引。2018—2019仅用于预定概率校准及告警规则，2020—2024用于一次正式评价。", "",
             "strict结果来自H的165格空间留出；temporal来自全域1580格。两个scope的分布和样本不同，分数差不等于纯迁移损失。", "",
             "## 固定复杂模型与最强简单对照", "",
             "ΔBS=LightGBM_F4−LR_F4，负值表示树更好。区间为预定14日配对层级bootstrap的95%百分位区间；改善年数按五个年度单列。", "",
             table(["范围", "lag", "风险集", "树BS", "LR BS", "ΔBS", "95%区间", "改善年数"], comparison_rows), "",
             "![固定F4比较](../figures/probability_phase4/fixed_f4_comparison.svg)", ""]
    for gate in gates:
        lines += [f"{gate['scope']} 的跨lag/风险组稳定优势及应用扩展保守门：**{'通过' if gate['passed'] else '未通过'}**。",
                  "全部逐cell判据与失败原因见 [application_gate.json](../data/probability_phase4/evaluation/application_gate.json)。整体门结论不抹去单项改善，也不构成部署许可。", ""]
    lines += ["## D−2新发生与湿度反例", "",
              "onset风险集包含D−1未发生的全部格日（00及01）。下表保留纯湿度和状态条件增量；两者回答不同问题。验证期反例与正式期结果应同时解释，不凭次要正面结果改写主对照。", "",
              table(["范围", "对照", "ΔBS", "95%区间", "改善年数"], moisture_rows), "",
              "![D−2新发生逐年湿度增量](../figures/probability_phase4/lag2_onset_moisture_years.svg)", "",
              "## 校准及基线", ""]
    statuses = pd.Series([r["mapping"]["status"] for r in calibrators["models"]]).value_counts().to_dict()
    lines += [f"104份映射记录状态：`{json.dumps(statuses, ensure_ascii=False)}`。持续性保留原始0/1，不参与概率映射。校准期诊断为样本内结果。", ""]
    baseline_rows = []
    for scope in ("temporal", "strict"):
        for lag in (1, 2):
            for model in ("persistence", "seasonal", "seasonal_pooled", "transition", "logistic_F0", "logistic_F4", "lightgbm_F4"):
                rows = choose(data["aggregate_scores"], scope=scope, lag=lag, model_id=model, risk_set="all")
                raw = rows[rows.probability.eq("raw")].iloc[0]
                calibrated = rows[rows.probability.eq("calibrated")].iloc[0]
                baseline_rows.append([scope, lag, model, number(raw.brier), number(calibrated.brier), number(raw.ap)])
    lines += [table(["范围", "lag", "配置", "原始BS", "校准BS", "原始AP"], baseline_rows), "",
              "其余全部特征臂、单类转换、子区、面积权重、7/21日敏感性及逐次删年结果均在机器表中保留；上述展示清单在开封前固定，并非测试最优排序。", "",
              "## 逐年可靠性", "",
              "每图五年独立显示固定十箱。空箱留空，不插值；计数、事件率和年度等权汇总见可靠性CSV。", ""]
    for scope in ("temporal", "strict"):
        for lag in (1, 2):
            lines += [f"![{scope} D−{lag}逐年可靠性](../figures/probability_phase4/reliability_{scope}_lag{lag}.svg)", ""]
    lines += ["## 可复现文件与边界", "",
              "- [冻结协议](PROBABILITY_PHASE4_PROTOCOL.md)、[复现命令](PROBABILITY_PHASE4_REPRODUCE.md)、[模型说明](PROBABILITY_MODEL_CARD.md)。",
              "- [逐年分数](../data/probability_phase4/evaluation/annual_scores.csv)、[汇总分数](../data/probability_phase4/evaluation/aggregate_scores.csv)、[配对差](../data/probability_phase4/evaluation/comparisons.csv)。",
              "- [配对区间](../data/probability_phase4/evaluation/bootstrap_intervals.csv)、[逐次删年](../data/probability_phase4/evaluation/leave_one_year_out.csv)、[逐年可靠性](../data/probability_phase4/evaluation/reliability.csv)。",
              "- [告警汇总](../data/probability_phase4/evaluation/aggregate_alert_metrics.csv)、[实际运行收据](../data/probability_phase4/evaluation/run_summary.json)。", "",
              "逐样本raw/calibrated预测保留在本地NPZ，连接年度时间/网格字典即可恢复含时区、lag及可用时间的明细。大数组不进入轻量Git备份，产物哈希用于复现核验。", "",
              "只有五个正式测试年份；2000次重采样不能创造独立年份。区间条件于已冻结模型与校准参数，单一H的时间区间不是跨未知地区的空间置信区间。", "",
              "此前工作接触过2017/2022案例，本轮不能称全新盲测。ERA5同化与发布限制仍在，D−2不保证消除未来信息。ESG用途仅为气候危害筛查和适应信息，不能替代厂内热暴露、职业安全标准或财务损失模型。", ""]
    report = Path("docs/PROBABILITY_PHASE4_RESULTS.md")
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    receipt = {"source_code_sha256": sha256(__file__),
               "input_sha256": {str(out / (n + '.csv')): sha256(out / (n + '.csv')) for n in filenames},
               "output_sha256": {str(path): sha256(path) for path in [report, *saved]},
               "purpose": "presentation_only_of_completed_frozen_evaluation"}
    (a.root / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(report), "figures": len(saved)}))


if __name__ == "__main__":
    main()
