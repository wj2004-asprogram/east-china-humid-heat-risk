"""Summarize already completed phase-2 artifacts, without fitting or new data."""

import argparse
import json
from pathlib import Path

import pandas as pd

from tools.probability_phase2_features import dump, sha256


def table(frame):
    headings = list(frame.columns)
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"]*len(headings)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def plot_moisture_deltas(yearly, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    selected = yearly[yearly.split.eq("validation")]
    for column, scope in enumerate(("temporal", "strict")):
        for row, risk in enumerate(("all", "onset")):
            axis = axes[row, column]
            for offset, lag, color in ((-.18, 1, "#2864a0"), (.18, 2, "#d47a25")):
                part = selected[selected.scope.eq(scope) & selected.lag.eq(lag) & selected.risk_set.eq(risk)]
                wide = part.pivot(index="year", columns="model", values="brier")
                if {"logistic_F0", "logistic_F1"}.issubset(wide.columns):
                    delta = wide.logistic_F1-wide.logistic_F0
                    axis.bar(np.arange(len(delta))+offset, delta, width=.34, label=f"D-{lag}", color=color)
            axis.axhline(0, color="#333333", lw=.9)
            axis.set_title(f"{'Temporal' if scope == 'temporal' else 'Strict training space'} / {'all samples' if risk == 'all' else 'onset risk set'}")
            axis.set_xticks([0, 1, 2], ["2015", "2016", "2017"])
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            if column == 0:
                axis.set_ylabel("F1 - F0 Brier (negative is better)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Humidity contrast: validation years already used for F4 C selection", fontsize=12)
    figure.text(.5, .015, "Shared finite sample keys. Onset includes 00 and 01. Raw probabilities; no confidence intervals.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .035, 1, .96))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/probability_phase2"))
    parser.add_argument("--experiment", default="run1")
    parser.add_argument("--report", type=Path, default=Path("docs/PROBABILITY_PHASE2_RESULTS.md"))
    args = parser.parse_args(argv)
    root, experiment = args.phase2_dir, args.phase2_dir/"experiments"/args.experiment
    runs = [json.loads((root/"features"/str(year)/"run.json").read_text()) for year in range(2000, 2020)]
    rows = []
    for run in runs:
        for scope, counts in run["counts"].items():
            rows.append({"year": run["year"], "scope": scope, **counts,
                         "runtime_seconds": run["runtime_seconds"], "peak_working_set_bytes": run["peak_working_set_bytes"]})
    pd.DataFrame(rows).to_csv(root/"feature_counts_yearly.csv", index=False)
    all_inputs = {path: digest for run in runs for path, digest in run["raw_inputs_sha256"].items()}
    if len(all_inputs) != 80 or any(str(year) in path.split("era5_east_china_")[-1][:4] for path in all_inputs for year in range(2020, 2025)):
        raise ValueError("Unexpected raw input year coverage")
    dump(root/"input_manifest.json", {"raw_inputs": all_inputs,
                                      "feature_runs": {str(run["year"]): sha256(root/"features"/str(run["year"])/"run.json") for run in runs}})
    yearly = pd.read_csv(experiment/"metrics_yearly.csv")
    summary = pd.read_csv(experiment/"metrics_summary.csv")
    fit_run = json.loads((experiment/"run.json").read_text())
    validation = summary[summary.split.eq("validation")]
    eligible = ["persistence", "seasonal", "transition", "logistic_F4", "logistic_F0"]
    champs = []
    for (scope, lag), part in validation[validation.risk_set.eq("all") & validation.model.isin(eligible)].groupby(["scope", "lag"]):
        winner = part.sort_values(["mean_annual_brier", "model"]).iloc[0]
        champs.append({"scope": scope, "lag": int(lag), "validation_champion": winner.model,
                       "mean_annual_raw_brier": float(winner.mean_annual_brier)})
    dump(experiment/"validation_champions.json", {"criterion": "mean annual raw Brier, 2015-2017",
         "eligible": eligible, "test_reselection": False, "champions": champs})
    contrasts = []
    for (scope, lag, risk), part in yearly[yearly.split.eq("validation")].groupby(["scope", "lag", "risk_set"]):
        by_model = part.pivot(index="year", columns="model", values="brier")
        if {"logistic_F0", "logistic_F1"}.issubset(by_model.columns):
            delta = by_model.logistic_F1-by_model.logistic_F0
            contrasts.append({"scope": scope, "lag": int(lag), "risk_set": risk,
                              "mean_delta_F1_minus_F0": float(delta.mean()),
                              "years_improving": int((delta < 0).sum()), "validation_years": len(delta),
                              "annual_delta": {str(year): float(value) for year, value in delta.items()}})
    dump(experiment/"moisture_contrasts.json", {"status": "validation_only_no_independent_test_claim", "contrasts": contrasts})
    plot_moisture_deltas(yearly, Path("figures/probability_phase2/moisture_validation_deltas.png"))
    selections = {scope: json.loads((experiment/f"lr_selection_{scope}.json").read_text()) for scope in ("temporal", "strict")}
    experiments = json.loads((experiment/"experiments.json").read_text())
    candidates = [candidate for selection in selections.values() for candidate in selection["candidates"]]
    reused = [item for item in experiments if item["model"].startswith("logistic_")
              and not (item["model"] == "logistic_F4" and item["lag"] == 1)]
    attempt_summary = {"search_attempts": len(candidates),
                       "search_converged": sum(item["converged"] for item in candidates),
                       "search_failed": sum(not item["converged"] for item in candidates),
                       "total_lr_attempts": len(candidates)+len(reused),
                       "total_lr_converged": sum(item["converged"] for item in candidates)+sum(item["converged"] for item in reused),
                       "exported_lr_models": sum(item["model"].startswith("logistic_") and item["status"] == "complete" for item in experiments)}
    attempt_summary["total_lr_failed"] = attempt_summary["total_lr_attempts"]-attempt_summary["total_lr_converged"]
    dump(experiment/"fit_attempt_summary.json", attempt_summary)
    failed_reuse = [f"{item['scope']}/{item['model']}/D−{item['lag']}" for item in reused if not item["converged"]]
    reuse_note = ("固定参数复用中未收敛、未导出预测的模型："+"；".join(failed_reuse)+"。" if failed_reuse else
                  "选定参数后的全部固定复用臂均收敛；搜索失败候选未混入导出的模型和概率。")
    failed_candidates = [f"{scope}: C={candidate['C']}"
                         for scope, selection in selections.items() for candidate in selection["candidates"]
                         if not candidate["converged"]]
    convergence_note = ("达到预设迭代上限而未收敛、已排除选择的候选："+"；".join(failed_candidates)+
                        "。按冻结规则保留失败记录，未增加迭代或更换求解器。" if failed_candidates else
                        "两个范围的全部有限搜索候选均按预设条件收敛。")
    by_contrast = {(row["scope"], row["lag"], row["risk_set"]): row for row in contrasts}
    temporal_d2_onset = by_contrast.get(("temporal", 2, "onset"))
    if temporal_d2_onset:
        moisture_summary = (f"需要保留的限制：普通范围D−2新发生风险集的F1−F0平均Brier差为"
                            f"{temporal_d2_onset['mean_delta_F1_minus_F0']:+.4f}，改善年份为"
                            f"{temporal_d2_onset['years_improving']}/3。在本轮固定LR设置和未校准验证概率下，当前结果不支持湿度稳定改善新发生；D−1的改善不能抵消D−2反例，总体分数改善也不能直接推广为新发生识别的稳定收益。")
    else:
        moisture_summary = "当前没有完整的普通范围D−2新发生湿度对照，不能对此作收益结论。"
    test_log = (root/"validation/targeted-final.log").read_text(encoding="utf-8", errors="replace")
    test_summary = next((line for line in reversed(test_log.splitlines()) if "passed" in line), "See validation log")
    body = ["# 湿热概率历史回报：第二阶段特征与真实基线结果", "",
            "本阶段完成2000–2019年完整特征和共同样本，以及2000–2014训练、2015–2017验证的有限真实基线。2018–2019只准备特征，未校准或评价；未打开2020–2024天气数值。以下为原始概率结果，训练指标是回代成绩，验证指标不能替代最终独立测试。", "",
            "## 特征与样本", "",
            "117列冻结特征已完成：C6/T27/M54/NT3/NM2/RT4/RM8/H13；F0=33、F1=87、F4=117列。真实小时均露点、Td/RH极值、12点夜间及20→07温差、3/7日汇总、季节背景和固定邻域均来自过去信息。HI标签由原计算精度的日HI生成，与特征矩阵float32存储转换独立；NetCDF保留各变量原计算dtype（日统计float32、夜间数组float64），无额外存储降精度。", "",
            "每年普通时间范围132,720个共同样本，严格训练空间109,116个；两lag及全部模型使用各范围同一键。20年均无额外缺测剔除。严格空间仅1299个训练查询格，邻域来源保持1302个训练空间格，不递归删除来源。", "",
            f"2000试跑{runs[0]['runtime_seconds']:.1f}秒，峰值工作集{runs[0]['peak_working_set_bytes']/2**20:.1f}MiB；20个正式年度提取共{sum(run['runtime_seconds'] for run in runs):.1f}秒。源文件共80个，均只读。", "",
            "## 有限LR选择", "",
            "每个范围仅F4/D−1比较C=0.01、0.1、1、10，使用2015/2016/2017逐年等权原始Brier选择；固定lbfgs、L2、不惩罚截距、max_iter=2000、tol=1e−6、2线程。选定C复用于F0/F1/F4及D−2，scaler只在训练样本拟合。", "",
            f"普通时间范围C={selections['temporal']['selected_C']}；严格训练空间C={selections['strict']['selected_C']}。候选收敛记录和逐年选择分数见各范围`lr_selection_*.json`。", "",
            convergence_note, "",
            f"实际搜索拟合{attempt_summary['search_attempts']}次：{attempt_summary['search_converged']}次收敛、{attempt_summary['search_failed']}次失败；连同固定参数复用共{attempt_summary['total_lr_attempts']}次真实LR拟合，{attempt_summary['total_lr_converged']}次收敛、{attempt_summary['total_lr_failed']}次失败，导出{attempt_summary['exported_lr_models']}个LR模型。", "",
            reuse_note, "",
            "## 验证原始概率", "",
            "Brier越低越好，AP越高越好。表中均为三个验证年等权均值；普通与严格范围样本不同，分数只在各自范围内比较。完整逐年训练/验证、混淆矩阵、事件率、余弦纬度敏感性与可靠度分箱见CSV。", ""]
    for risk, title in (("all", "全部共同样本"), ("onset", "新发生风险集：前日未发生，含负例00和正例01")):
        body += [f"### {title}", "", table(validation[validation.risk_set.eq(risk)][
            ["scope", "lag", "model", "mean_annual_brier", "mean_annual_ap"]]), ""]
    body += ["## 湿度对照与适用界限", "",
             "![逐年验证湿度对照](../figures/probability_phase2/moisture_validation_deltas.png)", "",
             table(pd.DataFrame(contrasts).query("risk_set in ['all', 'onset']")[["scope", "lag", "risk_set", "mean_delta_F1_minus_F0", "years_improving", "validation_years"]]), "",
             moisture_summary, "",
             "F1−F0差值仅描述本轮验证上的湿度对照，负值表示Brier改善。相同验证年已用于F4的C选择，不是未见评估；此处未计算置信区间或宣称显著性，也不解释成独立测试支持或因果机制。严格空间验证仍在训练空间，尚未检验留出地区迁移。D−2分组所用D−1标签仅供事后评价。当前p≥0.5只是固定研究阈值，未生成10%校准预算阈值。", "",
             "## 计算与证据", "", f"模型运行总计{fit_run['runtime_seconds']:.1f}秒，进程峰值工作集{fit_run['peak_working_set_bytes']/2**30:.2f}GiB。{test_summary}", "",
             "- 年度特征及SHA：`data/probability_phase2/features/YYYY/run.json`、`feature_counts_yearly.csv`、`input_manifest.json`。",
             f"- 指标与模型：`{experiment.as_posix()}/metrics_yearly.csv`、`metrics_summary.csv`、`models/`、`experiments.json`。",
             "- 预测采用每年NPZ与不可变网格表的因子化表示；`prediction_schema.json`给出样本ID、起报/目标窗、最大输入时间和缺失校准字段的重建规则。",
             "- 未提交大数组或隔离依赖；版本化JSON/CSV、输入和输出SHA以及重现命令保留。",
             "- 首次2000运行在写资源报告时遇到Windows ctypes句柄错误，数值产物保留在`2000-r1-incomplete`；修复后正式2000重跑成功，其后数值流程未变。", "",
             "下一阶段先完成尚未执行的预设模型与消融选择并冻结，再依协议进行概率校准、告警规则及最终测试。本轮未运行LightGBM或F0P/F1P等剩余对照，不据本报告直接开放最终测试。", ""]
    args.report.write_text("\n".join(body), encoding="utf-8", newline="\n")
    print(str(args.report))


if __name__ == "__main__":
    main()
