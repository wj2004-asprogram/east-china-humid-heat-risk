"""Report completed LightGBM batches and explicit pending comparisons, no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.probability_phase2_features import dump, sha256


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("NA" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def comparisons(tree, lr):
    """Keep unfinished pairs visible; conclusions require both lags in the arm."""
    left, right = tree.copy(), lr.copy()
    left["arm"] = left.model.str.removeprefix("lightgbm_")
    right["arm"] = right.model.str.removeprefix("logistic_")
    keys = ["scope", "lag", "arm", "year", "risk_set"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate model/year/risk-set records")
    merged = left.merge(right, on=keys, suffixes=("_tree", "_lr"), how="left", validate="one_to_one")
    available = merged.brier_lr.notna()
    if not ((merged.loc[available, "n_tree"] == merged.loc[available, "n_lr"]).all() and
            (merged.loc[available, "events_tree"] == merged.loc[available, "events_lr"]).all()):
        raise ValueError("Paired models do not share sample/outcome counts")
    merged["delta_brier_tree_minus_lr"] = merged.brier_tree-merged.brier_lr
    merged["bss_tree_vs_same_arm_lr"] = 1-merged.brier_tree/merged.brier_lr.where(merged.brier_lr.ne(0))
    tree_keys = set(zip(left.scope, left.lag, left.arm))
    lr_keys = set(zip(right.scope, right.lag, right.arm))
    merged["both_lags_available"] = [all((scope, lag, arm) in tree_keys and (scope, lag, arm) in lr_keys for lag in (1, 2))
                                     for scope, arm in zip(merged.scope, merged.arm)]
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/probability_phase3/lightgbm/run1"))
    parser.add_argument("--lr-dir", type=Path, default=Path("data/probability_phase3/lr_ablations/run1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/probability_phase2/experiments/run1"))
    parser.add_argument("--config", type=Path, default=Path("configs/probability_phase3_run.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/PROBABILITY_PHASE3_LIGHTGBM_RESULTS.md"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    registry = json.loads((args.run_dir/"execution_registry.json").read_text())
    output = args.run_dir/"report"
    output.mkdir(exist_ok=True)
    rows = []
    for entry in registry["fits"]:
        if entry["operation"] != "search" or entry["status"] == "not_executed":
            continue
        path = Path(entry["path"])
        result = json.loads(path.read_text()) if path.exists() else {"status": entry["status"]}
        rows.append({"scope": entry["scope"], **entry["candidate"], "status": result["status"],
                     "best_iteration": result.get("best_iteration"), "optimization_rounds": result.get("optimization_rounds"),
                     "fit_seconds": result.get("fit_seconds"), "validation_brier": result.get("mean_annual_validation_brier")})
    candidates = pd.DataFrame(rows)
    candidates.to_csv(output/"search_candidates.csv", index=False)
    pending = [entry for entry in registry["models"] if entry["status"] != "complete"]
    pd.DataFrame(pending, columns=["scope", "lag", "arm", "status", "path"]).to_csv(output/"pending_models.csv", index=False)
    selection_info = {}
    for scope in config["input"]["scopes"]:
        path = args.run_dir/f"selection_{scope}.json"
        if path.exists():
            winner = json.loads(path.read_text())["winner"]
            selection_info[scope] = {name: winner[name] for name in ("num_leaves", "min_child_samples", "reg_lambda", "best_iteration", "mean_annual_validation_brier")}
    dump(output/"selected_configurations.json", selection_info)
    model_metrics = args.run_dir/"metrics_yearly.csv"
    ablations_text = "尚无完整双 lag 的夜间、区域与历史事件增量对照。"
    comparisons_text = "当前尚无完整导出模型；候选验证概率仅用于冻结搜索，不替代完整同臂报告。"
    contrast_text = "尚无完成双lag的预设特征对照。"
    if model_metrics.exists():
        tree = pd.read_csv(model_metrics)
        tree = tree[tree.split.eq("validation")]
        if not set(tree.year).issubset({2015, 2016, 2017}):
            raise ValueError("Unexpected evaluation years")
        base = pd.read_csv(args.phase2_dir/"metrics_yearly.csv")
        simple = base[~base.model.str.startswith("logistic_") & base.split.eq("validation")]
        base_pairs = tree.merge(simple, on=["scope", "lag", "year", "risk_set"], suffixes=("_tree", "_baseline"))
        if not ((base_pairs.n_tree == base_pairs.n_baseline).all() and (base_pairs.events_tree == base_pairs.events_baseline).all()):
            raise ValueError("Simple baseline sample counts differ")
        base_pairs["delta_brier"] = base_pairs.brier_tree-base_pairs.brier_baseline
        base_pairs["bss_tree_vs_baseline"] = 1-base_pairs.brier_tree/base_pairs.brier_baseline.where(base_pairs.brier_baseline.ne(0))
        base_pairs.to_csv(output/"simple_baseline_pairs_yearly.csv", index=False)
        base_summary = base_pairs.groupby(["scope", "lag", "model_tree", "model_baseline", "risk_set"]).agg(
            years=("year", "count"), tree_brier=("brier_tree", "mean"), reference_brier=("brier_baseline", "mean"),
            delta_brier=("delta_brier", "mean")).reset_index()
        base_summary["bss_from_mean_annual_brier"] = 1-base_summary.tree_brier/base_summary.reference_brier.where(base_summary.reference_brier.ne(0))
        base_summary.to_csv(output/"simple_baseline_summary.csv", index=False)
        lr_parts = [base[base.model.str.startswith("logistic_") & base.split.eq("validation")]]
        if (args.lr_dir/"metrics_yearly.csv").exists():
            extra = pd.read_csv(args.lr_dir/"metrics_yearly.csv")
            lr_parts.append(extra[extra.split.eq("validation")])
        lr = pd.concat(lr_parts, ignore_index=True)
        # Supplement only: selection uses all-risk annual BS, never onset scores.
        champions = []
        for (scope, lag), part in lr[lr.risk_set.eq("all")].groupby(["scope", "lag"]):
            ranked = []
            for model, values in part.groupby("model"):
                arm = model.removeprefix("logistic_")
                if arm in config["features"]["arms"] and set(values.year) == {2015, 2016, 2017} and len(values) == 3:
                    ranked.append({"scope": scope, "lag": int(lag), "arm": arm,
                                   "mean_annual_brier": float(values.brier.mean()),
                                   "n_features": config["features"]["arms"][arm]["n_features"]})
            if ranked:
                winner = min(ranked, key=lambda row: (row["mean_annual_brier"], row["n_features"], row["arm"]))
                winner["completed_candidate_count"] = len(ranked)
                winner["full_planned_lr_set_complete"] = len(ranked) == len(config["features"]["arms"])
                champions.append(winner)
        dump(output/"supplementary_lr_champions.json", {
            "selection_bias": "Selected on already-used validation years; supplementary, not independent evidence",
            "rule": "all-risk equal-annual Brier; ties fewer features then lexicographic arm",
            "fixed_phase2_lr_f4_reference_unchanged": True, "selections": champions})
        if champions:
            selected_lr = pd.DataFrame(champions)[["scope", "lag", "arm"]]
            selected_lr["model"] = "logistic_"+selected_lr.pop("arm")
            reference = lr.merge(selected_lr, on=["scope", "lag", "model"], validate="many_to_one")
            supplementary = tree.merge(reference, on=["scope", "lag", "year", "risk_set"],
                                       suffixes=("_tree", "_selected_lr"), validate="many_to_one")
            supplementary["delta_brier"] = supplementary.brier_tree-supplementary.brier_selected_lr
            supplementary["bss_tree_vs_selected_lr"] = 1-supplementary.brier_tree/supplementary.brier_selected_lr.where(supplementary.brier_selected_lr.ne(0))
            supplementary.to_csv(output/"supplementary_lr_champion_pairs_yearly.csv", index=False)
        paired = comparisons(tree, lr)
        paired.to_csv(output/"same_arm_lr_pairs_yearly.csv", index=False)
        summary = paired.groupby(["scope", "lag", "arm", "risk_set", "both_lags_available"], dropna=False).agg(
            years=("year", "count"), tree_brier=("brier_tree", "mean"), lr_brier=("brier_lr", "mean"),
            delta_brier=("delta_brier_tree_minus_lr", "mean")).reset_index()
        summary["bss_from_mean_annual_brier"] = 1-summary.tree_brier/summary.lr_brier.where(summary.lr_brier.ne(0))
        summary.to_csv(output/"same_arm_lr_summary.csv", index=False)
        core = summary[summary.arm.isin(config["resources"]["core_arms_first"]) & summary.risk_set.isin(["all", "onset"])]
        comparisons_text = markdown_table(core[["scope", "lag", "arm", "risk_set", "both_lags_available", "tree_brier", "lr_brier", "delta_brier", "bss_from_mean_annual_brier"]])
        fixed_reference = lr[lr.arm.eq("F4")] if "arm" in lr else lr[lr.model.eq("logistic_F4")]
        fixed = tree.merge(fixed_reference, on=["scope", "lag", "year", "risk_set"], suffixes=("_tree", "_lrF4"), validate="many_to_one")
        fixed["delta_brier_tree_minus_phase2_LR_F4"] = fixed.brier_tree-fixed.brier_lrF4
        fixed["bss_tree_vs_phase2_lr_f4"] = 1-fixed.brier_tree/fixed.brier_lrF4.where(fixed.brier_lrF4.ne(0))
        fixed.to_csv(output/"fixed_phase2_lr_f4_pairs_yearly.csv", index=False)
        fixed_summary = fixed.groupby(["scope", "lag", "model_tree", "risk_set"]).agg(
            years=("year", "count"), tree_brier=("brier_tree", "mean"),
            reference_brier=("brier_lrF4", "mean"), delta_brier=("delta_brier_tree_minus_phase2_LR_F4", "mean")).reset_index()
        fixed_summary["bss_from_mean_annual_brier"] = 1-fixed_summary.tree_brier/fixed_summary.reference_brier.where(fixed_summary.reference_brier.ne(0))
        fixed_summary.to_csv(output/"fixed_phase2_lr_f4_summary.csv", index=False)
        contrast_rows = []
        present = set(zip(tree.scope, tree.lag, tree.model.str.removeprefix("lightgbm_")))
        for scope in config["input"]["scopes"]:
            for contrast in config["comparisons"]["predefined_feature_contrasts"]:
                candidate, reference = contrast["candidate"], contrast["reference"]
                complete = all((scope, lag, arm) in present for lag in (1, 2) for arm in (candidate, reference))
                for lag in (1, 2):
                    for risk in ("all", "onset", "persistence_or_clearance"):
                        part = tree[tree.scope.eq(scope) & tree.lag.eq(lag) & tree.risk_set.eq(risk)]
                        wide = part.pivot(index="year", columns="model", values="brier")
                        names = [f"lightgbm_{candidate}", f"lightgbm_{reference}"]
                        if all(name in wide for name in names):
                            delta = wide[names[0]]-wide[names[1]]
                            for year, value in delta.items():
                                contrast_rows.append({"scope": scope, "lag": lag, "risk_set": risk, "year": year,
                                    "candidate": candidate, "reference": reference, "both_lags_complete": complete,
                                    "delta_brier": value, "interpretation": contrast["interpretation"]})
        if contrast_rows:
            contrast_frame = pd.DataFrame(contrast_rows)
            contrast_frame.to_csv(output/"feature_contrasts_yearly.csv", index=False)
            contrast_summary = contrast_frame.groupby(["scope", "lag", "risk_set", "candidate", "reference", "both_lags_complete"]).agg(
                mean_delta=("delta_brier", "mean"), years_improving=("delta_brier", lambda values: int((values < 0).sum()))).reset_index()
            contrast_summary.to_csv(output/"feature_contrasts_summary.csv", index=False)
            completed_ablations = contrast_summary[contrast_summary.both_lags_complete &
                ~contrast_summary.candidate.isin(["F1", "F1P"]) & contrast_summary.risk_set.isin(["all", "onset"])]
            if not completed_ablations.empty:
                compact = completed_ablations.pivot(index=["scope", "candidate", "reference"],
                                                     columns=["lag", "risk_set"], values="mean_delta")
                compact.columns = [f"D{lag}_{risk}" for lag, risk in compact.columns]
                ablations_text = markdown_table(compact.reset_index())
            shown = contrast_summary[contrast_summary.candidate.isin(["F1", "F1P"]) & contrast_summary.risk_set.isin(["all", "onset"])]
            if not shown.empty:
                contrast_text = markdown_table(shown)
                d2_onset = shown[shown.lag.eq(2) & shown.risk_set.eq("onset") & shown.both_lags_complete]
                for row in d2_onset.itertuples():
                    contrast_text += (f"\n\n{row.scope} 的 D−2 新发生对照 {row.candidate}−{row.reference}："
                                      f"年度等权平均 Brier 差 {row.mean_delta:+.6f}，3 年中 {row.years_improving} 年改善。"
                                      "这一分组结果与全样本结果分别解释。")
    pilot_path = args.run_dir.parent/"pilot100/resource_pilot.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    invocations = [json.loads(path.read_text()) for path in sorted((args.run_dir/"invocations").glob("*.json"))]
    resource_batches = [json.loads(path.read_text()) for path in
                        (Path("data/probability_phase3/execution_batches.json"),
                         Path("data/probability_phase3/execution_batch2.json")) if path.exists()]
    metadata = {"status": "complete" if registry["exported_models"] == 44 else
                ("interrupted_partial" if registry.get("interrupted", 0) else "resource_batched_partial"),
                "run_config_sha256": sha256(args.config), "formal_registry": registry,
                "resource_pilot": pilot, "completed_invocations": invocations,
                "resource_batches": resource_batches,
                "selection": selection_info, "closed_years_read": False,
                "comparison_status": "Only completed same-arm, shared-key pairs; both-lag availability explicitly flagged"}
    dump(output/"report_state.json", metadata)
    title = "# 第三阶段 LightGBM：完整验证矩阵与消融结果" if registry["exported_models"] == 44 else "# 第三阶段 LightGBM：资源分批结果与待执行矩阵"
    body = [title, "",
            "本报告只汇总已完成批次。全部58次正式拟合、44个导出模型的目标不变；缺项明确列为pending，不根据分数删除实验。2018–2019及2020–2024均不读入本轮模型，不校准、不作最终测试或空间迁移声明。", "",
            f"当前正式完成/已拟合{registry['completed_or_fitted']}次，明确失败{registry['failed']}次，中断未完成{registry.get('interrupted', 0)}次，尚未执行{registry['not_executed']}次；已导出{registry['exported_models']}/44个模型。另有1次100轮训练期资源pilot，不计入正式58次。", "",
            "## 冻结方法与资源", "",
            "复用第二阶段117列float32与原始标签/共同样本，不重提取。CPU单线程、固定种子，训练期独立建分箱；验证Dataset以训练Dataset为reference。仅各范围F4/D−1的八个候选可用验证binary logloss早停100轮，最多2000轮；完成全候选终态后按逐年等权Brier及预设复杂度规则选定。所有其他臂和D−2复用本范围赢家参数及最佳轮数，不再早停或选轮数。", "",
            "首批树模型约3小时资源预算在首候选成绩出来前按pilot及迭代进度固定；达到预算不启动新拟合，当前拟合可结束。完整候选搜索未完成时不得声明赢家。严格树范围需要独立完整搜索，不能使用普通范围赢家代替。详细决定见`data/probability_phase3/execution_batches.json`。", ""]
    if len(resource_batches) > 1:
        second = resource_batches[1]["lightgbm"]
        body += [f"第二批为独立登记的资源调度：开始 {second['batch_start_utc']}，预算 {second['wall_budget_seconds']/3600:g} 小时，首批原截止不变。启动前确认磁盘与任务冲突门槛；复用已有结果，继续原预设矩阵，不按分数增删实验。记录见 `data/probability_phase3/execution_batch2.json`。", ""]
    if pilot:
        body += [f"100轮pilot：训练{pilot['training_rows']:,}行×117列，总{pilot['total_seconds']:.1f}秒，其中训练{pilot['fit_seconds']:.1f}秒，峰值工作集{pilot['peak_working_set_bytes']/2**30:.2f}GiB；没有验证输入或保留预测。", ""]
    historical_interruptions = list((args.run_dir/"interrupted_invocations").glob("*.json"))
    if historical_interruptions:
        body += [f"保留 {len(historical_interruptions)} 次历史执行中断记录（`run1/interrupted_invocations`）；恢复后完成结果覆盖当前pending状态，历史记录不删除。中断不是已证明的数值失败；工具返回标签本身不能确定用户停止意图。", ""]
    if registry.get("interrupted", 0):
        body += ["2026-11-05约20:15 UTC执行会话中断，等待工具返回 `aborted by user`；随后检查显示工作代理及Python训练进程均已结束。没有收到解释中断原因的新文字指令，因此不将原因猜测为数值错误或资源耗尽。strict第二候选最后日志为600轮、尚无训练完成检查点；保留为interrupted，后续需从该候选重训。第一候选及全部temporal产物可复用。第二资源批次尚未启动；本次交付为中断时的可恢复快照，不是严格空间完整结果。证据见 `run1/interrupted_invocations/20261105T201535Z.json`。", ""]
    body += ["## 已完成候选", "", markdown_table(candidates) if not candidates.empty else "尚无完成候选。", "",
             "选定参数（只列完整搜索的范围）：`report/selected_configurations.json`。训练轮数达到2000本身不等于失败；实际early-stopping最佳轮数与优化轮数分别保存。", "",
             "## 同臂 LR 配对", "", comparisons_text, "",
             "差值为LightGBM−LR，负值表示Brier降低。完整逐年、全部预设臂与固定第二阶段LR_F4参考见report内CSV。`both_lags_available=False`表示未完成双lag配对，只提供可追溯记录，不据此作完整对照结论。", "",
             "## 预设湿度对照", "", contrast_text, "",
             "所有结果来自已经用于选参的2015–2017验证年，没有独立测试或置信区间支持。F1−F0与F1P−F0P回答不同问题；历史事件状态已含湿度信息。第二阶段固定LR的D−2新发生反例仍保留，不能因树模型其他臂成绩较好而删除该证据。新发生风险集始终包含00和01，D−1标签在D−2中仅作事后评价。", "",
             "## 待执行与恢复", "",
             f"尚有{len(pending)}个模型未完整导出，逐项范围/lag/臂/状态见`report/pending_models.csv`与`execution_registry.json`。", "",
             "每个fit先保存模型文本及trained.json，再保存候选或逐年预测结果。恢复校验科学config、输入及参数指纹，已完成拟合不重训。初次搜索执行源快照已保存；后续资源守卫/调度补丁不改变科学参数，也不把已有模型作废。", "",
             "后续仍先完成预设选择与消融并冻结，再校准及最终测试；本阶段没有授权直接开放最终年份。", ""]
    body += ["Brier skill score (BSS) is 1 - BS_tree / BS_reference, with a zero reference score reported as NA. Annual tables use each year's scores; summary BSS uses the ratio of equal-weight mean annual scores, not the mean of annual BSS. Both same-arm LR and fixed phase-2 LR_F4 references are retained.", ""]
    body += ["年度原始指标表保留样本数、事件率、Brier、AP、TP/FP/FN/TN及可靠性分箱。概率未经校准；分类使用预定研究阈值 p≥0.5，没有用验证结果选择业务阈值或10%告警预算。训练指标属于训练样本内重代入结果。", ""]
    body += ["`report/simple_baseline_pairs_yearly.csv` 与汇总表保留相同范围、lag、年份和风险集下的持续性、季节气候概率及条件转移基线对照，包含 Brier 差和 BSS；这些基线直接复用第二阶段结果。", ""]
    body += ["## 其余预设增量与消融", "", ablations_text, "",
             "表中为候选减参照的年度等权平均 Brier 差，负值表示改善；只展示双 lag 均完成的对照。逐年变化、全部风险集和未完成状态保留在 report CSV 中，不由平均值推出稳定性或显著性。", ""]
    lr_run_path = args.lr_dir/"run.json"
    if lr_run_path.exists():
        lr_run = json.loads(lr_run_path.read_text())
        body += ["## LR 消融补齐", "",
                 f"本阶段新增 {lr_run['fits']} 次固定 C=0.1 的 LR 拟合，成功 {lr_run['successful_fits']} 次，失败 {lr_run['failed_fits']} 次。与第二阶段已有 12 个模型合并，覆盖两个范围、两个 lag 的全部 11 个特征臂。训练与缩放仅使用 2000–2014；strict 验证仍为 T 查询格，不代表空间迁移检验。", ""]
        contrasts_path = args.lr_dir/"lr_validation_contrasts.csv"
        if contrasts_path.exists():
            lr_contrasts = pd.read_csv(contrasts_path)
            shown_lr = lr_contrasts[lr_contrasts.candidate.eq("F1P") & lr_contrasts.reference.eq("F0P") & lr_contrasts.risk_set.isin(["all", "onset"])]
            body += ["加入最近允许事件状态后的湿度对照（F1P−F0P；负值表示改善）：", "", markdown_table(shown_lr), "",
                     "D−2 的新发生风险集在两个范围的平均 Brier 均变差；全样本改善不能替代新发生证据。该对照继续不支持湿度稳定改善 D−2 新发生预测。", ""]
        body += ["补充 LR 最优臂按全样本年度等权 Brier 一次选择，记录于 `report/supplementary_lr_champions.json`。这是已见验证集上的附加比较，存在选择偏差；主参照仍为同臂 LR 和固定第二阶段 LR_F4。", ""]
    if Path("figures/probability_phase3/moisture_validation_contrasts.png").exists():
        body += ["## 核心湿度对照图", "",
                 "![预设湿度对照](../figures/probability_phase3/moisture_validation_contrasts.png)", "",
                 "普通范围的两类预设对照、双 lag、LR 与 LightGBM，区分全样本和新发生风险集。负 Brier 差表示改善；这些验证年份已经参与模型选择，图中不作显著性推断。", "",
                 "[可导出 SVG](../figures/probability_phase3/moisture_validation_contrasts.svg)；逐年点、均值和输入指纹与图一同保存。", ""]
    if Path("figures/probability_phase3/feature_ladder_validation.png").exists():
        body += ["## 五层特征评分", "",
                 "![五层特征验证评分](../figures/probability_phase3/feature_ladder_validation.png)", "",
                 "普通范围 F0→F4 的 LR 与 LightGBM 验证评分，分别展示两个 lag 和全样本/新发生风险集。逐年点与年度等权均值均保留。", "",
                 "[可导出 SVG](../figures/probability_phase3/feature_ladder_validation.svg)；本图不代表独立测试或显著性检验。", ""]
    args.report.write_text("\n".join(body).replace("本轮树模型约3小时", "首批树模型约3小时"), encoding="utf-8", newline="\n")
    print(str(args.report), flush=True)


if __name__ == "__main__":
    main()
