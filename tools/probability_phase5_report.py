"""Build local site cards and figures from frozen, deduplicated evidence tables."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.probability_phase5_sites import read_json, sha, write_json


def clean(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=lambda x: x.item()).replace('NaN', 'null'))


def table_md(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def table_html(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+\
           '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def model_rows(card):
    return [[r['scope'], f"D−{r['lag']}", 'LR_F4' if r['model_id']=='logistic_F4' else 'LightGBM_F4',
             '全体' if r['risk_set']=='all' else '新发生风险集', r['n'], r['events'],
             f"{r['raw_brier']:.4f}", f"{r['calibrated_brier']:.4f}"] for r in card['model_summary']]


MODEL_HEADERS = ['范围', '输入', '冻结候选', '风险集', '格日', '事件日', '原始BS', '校准BS']


def figures(config, mapping, annual, scores, report):
    folder = report/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    registry = pd.read_csv(config['grid_registry'])
    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    for role, color in [('train','#b8c2cc'), ('holdout','#75afc5'), ('buffer','#dfba76')]:
        sub = registry[registry.spatial_role==role]
        ax.scatter(sub.longitude, sub.latitude, s=26, c=color, marker='s', label=f'Valid land: {role}')
    labels = {'factory_001':'Tesla (masked primary)', 'factory_002':'Ford', 'factory_003':'Ferrero',
              'factory_004':'Gree', 'factory_005':'ZF (buffer)'}
    offsets = {'factory_001':(10,-25), 'factory_002':(-75,25), 'factory_003':(-70,-28), 'factory_004':(16,-10), 'factory_005':(-100,15)}
    for row in mapping.itertuples():
        ax.plot([row.site_longitude,row.grid_longitude], [row.site_latitude,row.grid_latitude], c='#556', lw=.9)
        ax.scatter(row.site_longitude,row.site_latitude,c='#122c41',marker='^',s=50,zorder=4)
        ax.annotate(labels[row.site_id], (row.site_longitude,row.site_latitude), xytext=offsets[row.site_id],textcoords='offset points',fontsize=9)
    rejected = mapping[~mapping.temporal_available]
    ax.scatter(rejected.grid_longitude,rejected.grid_latitude,marker='x',s=100,color='#ad3a3a',label='Rejected primary')
    ax.set(xlim=(mapping.site_longitude.min()-.7,mapping.site_longitude.max()+.6),
           ylim=(mapping.site_latitude.min()-.7,mapping.site_latitude.max()+.6),
           xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',
           title='Archived site points and frozen spatial roles\nGrid centres; no coastline or exposure map')
    ax.legend(loc='lower right',fontsize=8)
    fig.savefig(folder/'mapping.png',dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4.8),layout='constrained')
    for grid, part in annual.groupby('grid_id'):
        ax.plot(part.year,part.days_ge_threshold,marker='o',ms=4,label=f'{grid} ({"3 sites" if mapping[mapping.primary_grid_id==grid].shape[0]==3 else "1 site"})')
    ax.set(xlabel='Year',ylabel='Days with daily maximum HI >= 40.6 C',ylim=(0,92),
           title='Two distinct primary-grid histories | 2000–2024 JJA\n92 days per summer; no claim of continuous factory operation')
    ax.grid(alpha=.18); ax.legend(fontsize=9)
    fig.savefig(folder/'historical_days.png',dpi=160); plt.close(fig)

    combos = scores[['scope','primary_grid_id']].drop_duplicates().sort_values(['scope','primary_grid_id']).values.tolist()
    for risk in ('all','onset'):
        fig, axes = plt.subplots(len(combos),len(config['lags']),figsize=(11,9),sharex=True,sharey=True,layout='constrained',squeeze=False)
        for i,(scope,grid) in enumerate(combos):
            for j,lag in enumerate(config['lags']):
                ax=axes[i,j]
                for model,color in [('logistic_F4','#176b7e'),('lightgbm_F4','#b95a31')]:
                    for prob,style in [('raw','-'),('calibrated','--')]:
                        part=scores[(scores.scope==scope)&(scores.primary_grid_id==grid)&(scores.lag==lag)&
                                    (scores.model_id==model)&(scores.probability==prob)&(scores.risk_set==risk)].sort_values('year')
                        ax.plot(part.year,part.brier,style,color=color,marker='o',ms=3,label=f'{model.replace("logistic","LR")} {prob}')
                ax.set_title(f'{scope} | {grid} | D−{lag}',fontsize=10)
                ax.grid(alpha=.2); ax.set_xticks(config['final_years'])
                if j==0: ax.set_ylabel('Brier score (lower is better)')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'Frozen candidates: {risk} | Annual descriptive scores\n84-day target window; 5 years, no site confidence interval')
        fig.savefig(folder/f'annual_{risk}.png',dpi=160); plt.close(fig)


def build(config_path, application_path):
    config, app = read_json(config_path), read_json(application_path)
    out, report = Path(config['output_root']), Path(config['report_root'])
    report.mkdir(parents=True,exist_ok=True); (report/'cards').mkdir(exist_ok=True)
    manifest = read_json(out/'extraction_manifest.json')
    for path, expected in manifest['outputs_sha256'].items():
        if sha(path)!=expected: raise ValueError(f'Extracted evidence changed: {path}')
    mapping = pd.read_csv(out/'site_mapping.csv')
    catalog = read_json(out/'source_catalog.json')
    sites = {s['site_id']:s for s in catalog['sites']}
    histories = {s['grid_id']:s for s in read_json(out/'historical_grid_summary.json')}
    scores = pd.read_csv(out/'aggregate_scores.csv')
    annual_scores = pd.read_csv(out/'annual_scores.csv')
    selected = scores[scores.model_id.isin(['logistic_F4','lightgbm_F4']) & scores.risk_set.isin(['all','onset'])]
    joined = selected.pivot(index=['scope','lag','model_id','primary_grid_id','risk_set','n','events'],columns='probability',values='brier').reset_index().rename(columns={'raw':'raw_brier','calibrated':'calibrated_brier'})
    cards, pathways, gaps, queue = [], [], [], []
    for row in mapping.to_dict('records'):
        sid, grid = row['site_id'], row['primary_grid_id']
        site, notes = sites[sid], app['sites'][sid]
        history = histories.get(grid)
        historical = None if history is None else {
            'period':config['historical_period'], 'complete_summers':history['complete_summers'],
            'mean_threshold_days':history['mean_days_ge_threshold_per_complete_summer'],
            'annual_days_distribution':history['distribution']['days_ge_threshold'],
            'max_consecutive_days':history['distribution']['max_consecutive_days']['max'],
            'selection':history['selection'],
            'warning':'Full 92-day JJA; not the 84-day hindcast window or a factory operating history.'}
        summary = joined[joined.primary_grid_id==grid].to_dict('records')
        card=clean(dict(site_id=sid,name=site['name'],entity=site['entity'],industry=site['industry'],
                        production_address=site['production_address'], snapshot_date=config['evidence_snapshot_date'],
                        verification_note=site['verification_note'], evidence_ids=site['evidence_ids'],
                        source_urls={key:catalog['sources'][key]['final_url'] for key in site['evidence_ids']},
                        mapping=row, historical=historical, model_summary=summary,
                        model_status='masked_primary_no_estimate' if not row['temporal_available'] else 'frozen_grid_hindcast_descriptive_only',
                        unknowns={key:None for key in app['unknown_fields']}, notes=notes,
                        impact_paths=app['pathways'], monitoring=app['common_monitoring_tasks'],limits=app['global_limits']))
        cards.append(card)
        pathways.extend(dict(site_id=sid,**p,industry_question=notes['industry_question']) for p in app['pathways'])
        gaps.extend(dict(site_id=sid,field=field,value=None,status='缺失现场数据',source='not_collected_no_imputation') for field in app['unknown_fields'])
        queue.append(dict(site_id=sid,queue=notes['priority'],suggestion=notes['next_action'],acceptance='可核查的边界、主体及观测覆盖记录；不改写冻结研究结果',status='待验证'))
        queue.extend(dict(site_id=sid,**q) for q in app['common_monitoring_tasks'])
    write_json(out/'site_cards.json',dict(protocol_id=config['protocol_id'],cards=cards))
    for name,rows in [('impact_pathways',pathways),('data_gaps',gaps),('monitoring_queue',queue)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,lineterminator='\n')
    annual = pd.read_csv(out/'historical_annual.csv')
    figures(config,mapping,annual,annual_scores,report)
    for card in cards:
        m,h = card['mapping'],card['historical']
        history_text = '主格被掩膜拒绝；历史阈值天数、连续日数和概率保持为空。缺值不表示低危害。' if h is None else (
            f"2000—2024的25个完整夏季，每季92日：日最大HI≥{config['threshold_c']}°C的天数均值 {h['mean_threshold_days']:.2f}日；"
            f"年度最小/中位/最大为 {h['annual_days_distribution']['min']:g}/{h['annual_days_distribution']['median']:g}/{h['annual_days_distribution']['max']:g}日；"
            f"季内最长连续 {h['max_consecutive_days']:g}日。原定典型年 {h['selection']['typical']['year']}，极端年 {h['selection']['extreme']['year']}。这些是格点背景，不是企业历史损失。")
        card['history_text']=history_text
        links_md='\n'.join(f'- [{key}]({url})' for key,url in card['source_urls'].items())
        model_md=table_md(MODEL_HEADERS,model_rows(card)) if card['model_summary'] else '没有主格模型回报；未提取北侧敏感格概率作为替代。'
        body=f"""# {card['name']} · 湿热危害证据卡

证据快照：{card['snapshot_date']}。用途：研究筛查与补证；不用于职业安全、停工或财务风险判断。

## 已有公开/气候证据

主体：{card['entity']}。生产地址：{card['production_address']}。行业：{card['industry']}。

{card['verification_note']}

原主格：{m['primary_grid_id']}；状态：{m['screening_status']}；空间角色：{m['spatial_role'] or '无有效陆地格'}；代表点距格心 {m['representative_distance_km']:.2f} km。

{card['notes']['representation']}

{history_text}

## 已冻结模型的描述性回报

2020—2024，每年6月9日—8月31日84日；五年等权BS。新发生风险集包括D−1未发生的全部日期（00及01），不是仅选事件日。D−2的D−1状态仅用于事后评价。temporal是跨年回报，strict仅为H留出模型；不可择优scope或将分数差当纯迁移损失。LR_F4是原验证期冻结候选，不是根据本卡成绩选择。

{model_md}

同格场所共享以上数值，不能增加气候样本。其余四个强基线、逐年指标、持续/解除组、十箱计数和冻结告警规则见[完整数据](../../../data/probability_phase5/aggregate_scores.csv)与[回报明细](../../../data/probability_phase5/predictions.csv)。单厂不生成置信区间；区域区间不能直接移植。研究告警阈值不构成现场行动标准。

## 合理但未验证路径

{card['notes']['industry_question']}

"""
        for p in card['impact_paths']:
            body+=f"- {p['path']}。需取得：{p['requires']}。{p['not_established']}\n"
        body+='\n## 缺失现场数据\n\n'+ '、'.join(card['unknowns'])+' 均为空；未用行业经验填补。\n\n## 待验证建议\n\n'+card['notes']['priority']+'\n\n'+card['notes']['next_action']+'\n\n'
        body+='\n\n'.join(q['suggestion']+' '+q['acceptance']+' '+q['status']+'。' for q in card['monitoring'])
        body+='\n\n这是补证工作队列，不是企业风险排名；没有开展现场联系、监测或采购。\n\n## 适用边界与来源\n\n'+'\n'.join('- '+s for s in card['limits'])+'\n\n'+links_md+'\n\n[返回研究报告](../index.html) · [来源哈希与检索日期](../../../data/probability_phase5/source_catalog.json)\n'
        (report/'cards'/f"{card['site_id']}.md").write_text(body,encoding='utf-8',newline='\n')
    write_json(out/'site_cards.json',dict(protocol_id=config['protocol_id'],cards=cards))
    summary_rows=[]
    for card in cards:
        m,h=card['mapping'],card['historical']
        summary_rows.append([card['name'],m['primary_grid_id'],m['spatial_role'] or '掩膜拒绝',
                             f"{h['mean_threshold_days']:.2f}" if h else '缺失',
                             f"{h['max_consecutive_days']:g}" if h else '缺失',
                             'temporal + strict H' if m['strict_H_available'] else ('temporal' if m['temporal_available'] else '无主格结果')])
    summary_headers=['场所','原主格','角色','历史均值日/92日','最长连续日','可用正式回报']
    selected_rows=joined[joined.model_id=='logistic_F4'].to_dict('records')
    lr_table=model_rows({'model_summary':selected_rows})
    # The grid column distinguishes temporal results at different sites.
    lr_table=[[r['primary_grid_id']]+row for r,row in zip(selected_rows,lr_table)]
    results=f"""# 五场所湿热危害筛查与研究交付

已完成五张证据卡、两条有效主格历史序列与冻结概率的描述性连接。**四个有效场所只有两个气候格点；特斯拉主格拒绝保留为空，杭州三厂共格，采埃孚只具temporal回报。** 本阶段没有训练、校准或使用正式测试重新选择模型。

## 数据与空间代表性

{table_md(summary_headers,summary_rows)}

原公开资料快照为2026-10-29，没有本轮现场验证或当前在营确认。几何100%落在一个最近中心分区不能证明厂内代表性；代表点距格心约8.7—13.9 km。特斯拉70.05%/29.95%的几何分区保留原掩膜，不以北侧有效格替代或重归一化。采埃孚同址两实体的边界覆盖仍待核实。

历史为2000—2024完整JJA、92日/年：4600去重格日、50格年。两个有效主格有407个季内连续过程，8个触及季边界；过程不假设独立。典型/极端年份沿用既有规则，没有本阶段再挑案例。正式回报为2020—2024每年84日，840个物理格日；strict杭州的420日复用同一观测，不增加样本。120个既有预测文件产生15120条模型记录，包含两个lag和六个固定候选/对照。

## 冻结LR_F4的场所格点回报

下表为五年等权BS（越小越好），n和事件数是五年总计。这里只描述已固定模型，不做场所择优或统计显著性结论。原始概率是主科学比较，校准结果完整并列；不能按组挑概率版本。

{table_md(['主格']+MODEL_HEADERS,lr_table)}

杭州主格的正式窗口共有266个阈值日、40个新发生日；采埃孚主格共有212个阈值日、44个新发生日。新发生风险集的分母分别157和210日，含00和01。两格的temporal D−2新发生LR校准BS均变差：杭州0.1502→0.1528，采埃孚0.1365→0.1402；strict杭州0.154273→0.154281也没有实质改善。五年变动和少量事件限制了单格结论，未套用区域置信区间。

全部四个简单基线、LightGBM_F4与LR_F4、两个lag、raw/cal、逐年和持续/解除风险组均在[描述表](../data/probability_phase5/aggregate_scores.csv)、[逐年表](../data/probability_phase5/annual_scores.csv)、[十箱计数](../data/probability_phase5/reliability.csv)与[固定告警表](../data/probability_phase5/annual_alerts.csv)。告警预算是原校准期规则，不能保证测试或当地比例≤10%，也不是企业行动阈值。

## 与研究结论的关系

[第四阶段结论](PROBABILITY_PHASE4_INTERPRETATION.md)仍有效：D−1湿度增量有本轮历史回报证据，但D−2新发生未支持稳定改善；LightGBM_F4没有通过替换LR_F4的稳定优势门。100个生产概率映射拟合完成不等于当地概率可靠；第四阶段4条事后截距/斜率诊断失败保留NA。单场所描述表不能推翻或重新筛选这些结论。

原机器诊断标签 `interval_does_not_exclude_zero` 的准确含义为“区间未完全低于零”；部分区间其实完全为正，支持树更差，仍不满足优势判据。数值与判定未变，详见[最终质量核查及标签勘误](PROBABILITY_PHASE5_FINAL_QUALITY_REVIEW.md)。

ERA5同化和可用时间限制、既有2017/2022背景接触与五个测试年的不确定性均保留；本阶段规则在第四阶段结果已知后落实，不称盲测或预注册。H是单一空间留出且使用允许时刻的近期本地气象输入；temporal/strict分数差不是纯迁移损失。

## 影响路径、数据缺口与下一步

每张卡区分已有证据、未验证路径、缺失数据和待验证建议。人员热负荷、冷却需求、生产/质量影响只是需要现场资料检验的路径；没有真实暴露、健康、运营异常、损失或企业风险估计。HI≥40.6°C不能转换为WBGT、停工或职业合规标准。

补证队列A优先解除定位、边界和覆盖问题：特斯拉掩膜/代表性；采埃孚同址生产单元；杭州三厂各自作业区和共同格背景。队列B建立同步厂内外观测及质量记录，队列C在取得合法访问后连接实际工况/能耗/异常资料。队列可并行，是信息任务的先后依赖，不是企业风险排名。所有现场建议标记待验证，没有现场联系、采购或实际监测。

未来应先取得独立观测和带起报/发布时间的真实预报，另立验证方案；2020—2024现为已接触资料，不作为新模型盲测。缺少现场边界确认、温湿度/WBGT、岗位班次、冷却设施与运营资料，是目前不能升级为企业应用的实质缺口。

## 可复现交付

- [本地报告与五卡](../reports/probability_phase5/index.html)
- [机器可读五卡](../data/probability_phase5/site_cards.json)、[去重概率明细](../data/probability_phase5/predictions.csv)、[映射表](../data/probability_phase5/site_mapping.csv)
- [路径](../data/probability_phase5/impact_pathways.csv)、[缺口](../data/probability_phase5/data_gaps.csv)、[监测补证队列](../data/probability_phase5/monitoring_queue.csv)
- [模型使用补充](PROBABILITY_PHASE5_MODEL_USE.md)、[复现入口](PROBABILITY_PHASE5_REPRODUCE.md)、[协议](PROBABILITY_PHASE5_PROTOCOL.md)
- [提取验证](../data/probability_phase5/validation.json)与[源/结果哈希](../data/probability_phase5/extraction_manifest.json)

原项目、旧界面及第四阶段冻结文件未改动；新增结果保存在独立研究分支。本阶段不生成新的训练模型，不部署、不上传原始数据。
"""
    Path('docs/PROBABILITY_PHASE5_RESULTS.md').write_text(results,encoding='utf-8',newline='\n')
    css='''body{margin:0;background:#f4f6f7;color:#183044;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:32px 22px 70px}header{padding:30px 0;border-bottom:3px solid #176b7e}h1{font-size:32px;line-height:1.3}h2{margin-top:1.8em;font-size:24px}h3{font-size:19px}p{max-width:100ch}a{color:#08627c}nav{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}.note{background:#fff1d6;border-left:5px solid #b2771e;padding:16px 20px}.panel{background:white;border:1px solid #d5e0e5;border-radius:10px;padding:22px;margin:22px 0}table{border-collapse:collapse;font-size:14px;width:100%}th,td{border-bottom:1px solid #d4dfe4;padding:9px;text-align:left;vertical-align:top}th{background:#e8f1f4}.table-scroll{overflow:auto}img{display:block;width:100%;height:auto;background:white}figure{margin:24px 0}figcaption{font-size:14px;color:#486070}.tag{font-size:13px;padding:3px 9px;border-radius:20px;background:#e5f1ef}details{padding:10px 0}summary{cursor:pointer;font-weight:600}code{font-size:13px;overflow-wrap:anywhere}footer{border-top:1px solid #c7d3db;margin-top:32px;padding-top:16px;color:#486070}@media(max-width:650px){main{padding:16px 12px}h1{font-size:26px}.panel{padding:14px}th,td{padding:6px;min-width:70px}}'''
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>五场所湿热危害证据 · 冻结研究交付</title><style>'+css+'</style></head><body><main>'
    page+='<header><span class="tag">阶段五 · 本地研究交付 · 公开资料快照 2026-10-29</span><h1>五个制造业场所的湿热危害证据</h1><p>把历史格点危害、冻结模型回报与现场信息缺口连接起来。五张卡对应两个有效气候主格；一个场所主格保留缺失。</p></header>'
    page+='<nav>'+''.join(f'<a href="#{c["site_id"]}">{escape(c["name"])}</a>' for c in cards)+'</nav>'
    page+='<div class="note"><strong>研究筛查，不构成企业风险判断。</strong> ERA5概率历史回报没有验证实时可用性。HI阈值不等于职业安全或损失标准；尚无现场观测、岗位、设备或运营记录。现场建议全部待验证。</div>'
    page+='<h2>先看可用证据与边界</h2><p>阈值事件：北京时间一天内，日最大HI≥40.6°C。窄屏可横向滚动表格。</p>'+table_html(summary_headers,summary_rows)
    page+='<p>完整历史：2000—2024，每夏92日，共4600格日。正式回报：2020—2024，每夏84日，共840个物理格日。杭州三厂共格，strict输出复用相同观测，均不增加独立样本。采埃孚在buffer，仅有temporal结果；特斯拉没有主格概率。</p>'
    page+='<figure><img src="figures/mapping.png" alt="场所代表点、原主格和冻结空间角色"><figcaption>冻结格点角色和存档代表点。没有海岸线或厂内暴露含义；几何匹配不证明实测代表性。</figcaption></figure>'
    page+='<figure><img src="figures/historical_days.png" alt="两个有效主格25年阈值天数"><figcaption>两个主格的25年完整夏季历史。共享气候背景不能区分杭州三厂暴露；历史资料不证明企业连续运营。</figcaption></figure>'
    page+='<h2>保留原研究结论与反例</h2><p>第四阶段没有支持用LightGBM_F4替换冻结LR_F4。D−1湿度增量有回报证据，但D−2新发生缺少稳定改善证据。校准可改善总体却恶化新发生：杭州temporal D−2的LR BS为0.1502→0.1528，采埃孚为0.1365→0.1402。这里不进行当地模型、scope或阈值择优。</p><p>新发生风险集含D−1无事件的全部日期，包含00与01；D−2使用D−1状态只作事后分组。原始/校准概率并列，区域置信区间不能作为单厂区间。</p>'
    for name,title in [('annual_all','全部目标日的逐年BS'),('annual_onset','新发生风险集的逐年BS')]:
        page+=f'<figure><img src="figures/{name}.png" alt="{title}"><figcaption>{title}。五个年份的描述性结果；没有新增单厂置信区间。temporal和strict分别展示，不将差值解释为纯空间迁移损失。</figcaption></figure>'
    page+='<h2>五张场所证据卡</h2>'
    for c in cards:
        status='主格拒绝 · 结果缺失' if c['historical'] is None else '可用格点回报 · 描述性证据'
        page+=f'<article class="panel" id="{c["site_id"]}"><span class="tag">{status}</span><h3>{escape(c["name"])}</h3><p>{escape(c["entity"])} · {escape(c["production_address"])}</p><p><strong>已有证据：</strong>{escape(c["verification_note"])}</p><p>{escape(c["notes"]["representation"])}</p><p>{escape(c["history_text"])}</p>'
        if c['model_summary']:
            page+='<details><summary>查看固定F4模型回报（五年等权；BS越小越好）</summary>'+table_html(MODEL_HEADERS,model_rows(c))+'</details>'
        page+='<p><strong>未验证路径：</strong>人员热负荷、冷却需求、潜在生产或质量影响均需现场证据。'+escape(c['notes']['industry_question'])+'</p>'
        page+='<p><strong>缺失数据：</strong>当前运营状态、岗位班次、实际温湿度/WBGT、风与辐射、防护装备、冷却设施、能耗、质量/停机和损失资料均未取得。</p>'
        page+='<p><strong>补证队列：</strong>'+escape(c['notes']['priority'])+'</p><p>'+escape(c['notes']['next_action'])+'</p>'
        page+=f'<p><a href="cards/{c["site_id"]}.md">完整证据卡与来源</a></p></article>'
    page+='<h2>下一步是补证工作，不是企业排名</h2><p>A：核清边界与代表性；B：具备许可后建立可追溯的厂内外同步观测；C：在取得实际运营资料后检验影响路径。三类任务可并行，但没有现场证据前不估算损失或适应收益。独立气象观测和真实起报预报需另立前瞻验证；已使用的2020—2024不能再包装成盲测。</p>'
    downloads=[('完整研究结果','../../docs/PROBABILITY_PHASE5_RESULTS.md'),('模型使用补充','../../docs/PROBABILITY_PHASE5_MODEL_USE.md'),('复现入口','../../docs/PROBABILITY_PHASE5_REPRODUCE.md'),('阶段协议','../../docs/PROBABILITY_PHASE5_PROTOCOL.md'),('区域正式结论','../../docs/PROBABILITY_PHASE4_INTERPRETATION.md'),('五卡JSON','../../data/probability_phase5/site_cards.json'),('预测明细CSV','../../data/probability_phase5/predictions.csv'),('全部描述指标','../../data/probability_phase5/aggregate_scores.csv'),('逐年指标','../../data/probability_phase5/annual_scores.csv'),('十箱计数','../../data/probability_phase5/reliability.csv'),('冻结告警指标','../../data/probability_phase5/annual_alerts.csv'),('来源目录','../../data/probability_phase5/source_catalog.json'),('影响路径','../../data/probability_phase5/impact_pathways.csv'),('数据缺口','../../data/probability_phase5/data_gaps.csv'),('监测补证队列','../../data/probability_phase5/monitoring_queue.csv'),('提取验证','../../data/probability_phase5/validation.json')]
    downloads.insert(0,('最终质量核查与标签勘误','../../docs/PROBABILITY_PHASE5_FINAL_QUALITY_REVIEW.md'))
    page+='<h2>资料与复现</h2><nav>'+''.join(f'<a href="{href}">{label}</a>' for label,href in downloads)+'</nav>'
    page+='<footer>离线静态报告；无需联网运行，无数据上传或外部脚本。原界面保持原样。2026-10-29公开来源快照不代表当前现场状态；公开来源链接见各卡。</footer></main></body></html>'
    (report/'index.html').write_text(page,encoding='utf-8',newline='\n')
    output_files=[out/'site_cards.json',out/'impact_pathways.csv',out/'data_gaps.csv',out/'monitoring_queue.csv',Path('docs/PROBABILITY_PHASE5_RESULTS.md'),report/'index.html']+sorted((report/'cards').glob('*.md'))+sorted((report/'figures').glob('*.png'))
    write_json(out/'report_manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_extraction_manifest_sha256=sha(out/'extraction_manifest.json'),
        configuration_sha256={str(p):sha(p) for p in [config_path,application_path]},
        source_code_sha256={'tools/probability_phase5_report.py':sha('tools/probability_phase5_report.py')},
        software={'matplotlib':matplotlib.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        outputs_sha256={p.as_posix():sha(p) for p in output_files},
        cards=len(cards), distinct_climate_sequences=len(histories), no_new_model_selection=True))
    print(f'Created {len(cards)} cards, {len(pathways)} conditional pathways, {len(gaps)} missing fields, {len(queue)} monitoring tasks, 4 figures and local report.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/probability_phase5_sites.json'))
    parser.add_argument('--application',type=Path,default=Path('configs/probability_phase5_application.json'))
    args=parser.parse_args()
    build(args.config,args.application)
