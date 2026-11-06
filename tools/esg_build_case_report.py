"""Build the five-site case report and standalone scientific figures from audited results."""
import argparse
import hashlib
import html
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#285b85', '#a66036']
LABELS = {'30.2500,120.5000': '杭州共享网格（3个场所）', '31.0000,121.5000': '采埃孚对应网格（1个场所）'}
SCENARIOS = {'accepted_primary':'原主代表点', 'laea_area_centroid':'面积质心', 'laea_point_on_surface':'多边形内部点', 'largest_polygon_overlap_grid':'最大覆盖网格'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_script(path):
    return json.loads(path.read_text(encoding='utf-8').split('=', 1)[1].strip().removesuffix(';'))


def check_manifest(directory, filename):
    manifest = load(directory/filename)
    for name, digest in manifest['output_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError(f'Changed analysis output: {name}')
    return manifest


def fmt(value, digits=0):
    return '—' if value is None else f'{value:,.{digits}f}'


def table(headers, rows):
    e = lambda value: html.escape(str(value))
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+'.svg'), bbox_inches='tight', metadata={'Date':None})
    fig.savefig(directory/(name+'.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


def figures(temporal, spatial, geometries, research, output):
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.unicode_minus':False, 'svg.hashsalt':'EHHR-FIVE-SITE-01'})
    output.mkdir(parents=True, exist_ok=True)
    valid = [g for g in temporal['grids'] if g['land_mask'] == 1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, constrained_layout=True)
    for ax, metric, label in zip(axes, ['days_ge_threshold','max_consecutive_days'], ['年度阈值天数 D / 天', '年度最长连续 L / 天']):
        for grid, color in zip(valid, COLORS):
            rows = grid['annual']
            ax.plot([a['year'] for a in rows], [a[metric] for a in rows], color=color, marker='o', ms=3, lw=1.4, label=LABELS[grid['grid_id']])
        ax.set_ylim(0, 92); ax.set_yticks([0,23,46,69,92]); ax.set_ylabel(label); ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, loc='upper left', ncols=2)
    axes[1].set_xticks([2000,2005,2010,2015,2020,2024]); axes[1].set_xlabel('年份（各年JJA分别统计）')
    save_figure(fig, output, 'annual-comparison')

    rg = {g['grid_id']:g for g in research['grids']}
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharey=True, constrained_layout=True)
    for row, grid in enumerate(valid):
        for col, role in enumerate(['typical','extreme']):
            year = grid['selection'][role]['year']
            days = [d for d in rg[grid['grid_id']]['daily'] if d['date'].startswith(str(year)+'-')]
            values = np.array([d['daily_max_heat_index_c'] for d in days])
            ax = axes[row,col]
            for i in np.flatnonzero(values >= temporal['threshold_c']):
                ax.axvspan(i-.5, i+.5, color='#ecd9b9', alpha=.45, linewidth=0)
            ax.plot(range(len(values)), values, lw=1.1, color=COLORS[row])
            ax.axhline(temporal['threshold_c'], color='#8d532f', ls='--', lw=.9)
            annual = next(a for a in grid['annual'] if a['year']==year)
            ax.set_title(f'{LABELS[grid["grid_id"]]} · {"典型年" if role=="typical" else "阈值天数最多年"} {year}\nD={annual["days_ge_threshold"]}天；L={annual["max_consecutive_days"]}天', fontsize=10)
            ax.set_xlim(-.5,91.5); ax.set_ylim(0,60); ax.set_yticks([0,20,40,60]); ax.grid(axis='y',alpha=.2)
            ax.set_xticks([0,30,61,91],['06-01','07-01','08-01','08-31'])
            if col==0: ax.set_ylabel('日最高 HI / °C')
    save_figure(fig, output, 'selected-summers')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, audit in zip(axes.flat, spatial['site_audits']):
        members = [f for f in geometries['features'] if f['properties']['site_id']==audit['site_id']]
        positions=[]
        for feature in members:
            geometry=feature['geometry']
            polys=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
            for rings in polys:
                positions.extend(rings[0])
                ax.add_patch(Polygon(rings[0], closed=True, facecolor='#cadfea' if feature['properties']['land_mask']==1 else '#dedee0', edgecolor='#466176', lw=.9))
                for hole in rings[1:]: ax.add_patch(Polygon(hole,closed=True,facecolor='white',edgecolor='#466176',lw=.8))
        x,y=np.array(positions).T
        pad=max(x.max()-x.min(),y.max()-y.min())*.13
        ax.set_xlim(x.min()-pad,x.max()+pad); ax.set_ylim(y.min()-pad,y.max()+pad)
        points=[p for p in spatial['representative_scenarios'] if p['site_id']==audit['site_id'] and p['longitude'] is not None]
        for p,marker,color in zip(points,['o','x','^'],['#a6452d','#2c6949','#222']):
            ax.scatter(p['longitude'],p['latitude'],marker=marker,color=color,s=37,zorder=4)
        ax.set_aspect(1/math.cos(math.radians(float(y.mean()))))
        ax.set_title(audit['name'],fontsize=11)
        ax.ticklabel_format(useOffset=False,style='plain'); ax.tick_params(labelsize=8)
        ax.set_xlabel('经度 / °E'); ax.set_ylabel('纬度 / °N')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3)); ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.95,'存档OSM多边形 × 最近中心分区\n\n浅蓝：对应格通过陆地掩膜\n灰色：对应格未通过陆地掩膜\n\n红圆：原主代表点\n绿叉：面积质心\n黑三角：多边形内部点\n\n各子图范围不同；不是同尺度比较。\n颜色不是厂区真实海陆性质。\n本图不包含影像或地籍核验。',va='top',fontsize=11,linespacing=1.8)
    save_figure(fig, output, 'polygon-grid-audit')


def figure_html(name, caption):
    return f'<figure><img src="figures/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption><a class="download-figure" href="figures/{name}.svg" download>下载SVG</a></figure>'


def build(temporal_dir, spatial_dir, research_path, catalog_path, protocol_path, output):
    tm=check_manifest(temporal_dir,'manifest.json'); sm=check_manifest(spatial_dir,'spatial_manifest.json')
    for manifest in (tm,sm):
        if manifest['input_sha256']['protocol'] != sha(protocol_path) or manifest['input_sha256']['research_data'] != sha(research_path):
            raise ValueError('Report inputs differ from audited module inputs')
    temporal=load(temporal_dir/'case_temporal.json'); spatial=load(spatial_dir/'spatial_audit.json')
    geometries=load(spatial_dir/'factory_grid_intersections.geojson'); research=read_script(research_path)
    catalog=load(catalog_path); protocol=load(protocol_path)
    if sha(catalog_path)!=tm['input_sha256']['catalog'] or sha(catalog_path)!=sm['input_sha256']['catalog']:
        raise ValueError('Catalog changed after analysis')
    output.mkdir(parents=True,exist_ok=True)
    for folder,source in [('temporal',temporal_dir),('spatial',spatial_dir)]:
        dest=output/'downloads'/folder;dest.mkdir(parents=True,exist_ok=True)
        for file in source.iterdir():
            if file.is_file():shutil.copyfile(file,dest/file.name)
    docs=['ESG_FIVE_SITE_PROTOCOL.md','ESG_FIVE_SITE_DATA_DICTIONARY.md','ESG_FIVE_SITE_METHOD_AUDIT.md','ESG_FIVE_SITE_RESULTS.md']
    for name in docs:shutil.copyfile(ROOT/'docs'/name,output/'downloads'/name)
    shutil.copyfile(protocol_path,output/'downloads/protocol.json')
    code=output/'downloads/code';code.mkdir(exist_ok=True)
    for name in ['esg_case_temporal.py','esg_case_spatial.py','esg_build_case_report.py']:
        shutil.copyfile(ROOT/'tools'/name,code/name)
    (output/'downloads/README.txt').write_text('本资料包含阶段1—3产物、固定方案、数据字典、方法核查和三个分析脚本。脚本应放回科学项目的tools目录运行，依赖现有项目模块与既有研究/气候输入。原始NetCDF与完整来源网页不在资料包内。输入输出哈希见temporal/manifest.json、spatial/spatial_manifest.json。CSV原精度保留；空值不是0。\n',encoding='utf-8',newline='\n')
    shutil.copyfile(ROOT/'apps/factory-pilot/case-report.css',output/'report.css')
    shutil.copyfile(ROOT/'apps/factory-pilot/case-report.js',output/'report.js')
    figures(temporal,spatial,geometries,research,output/'figures')
    valid=[g for g in temporal['grids'] if g['land_mask']==1]
    episodes=[e for g in valid for e in g['episodes']]
    payload={'grids':[{'grid_id':g['grid_id'],'label':LABELS[g['grid_id']]} for g in valid],'episodes':episodes}
    (output/'report-data.js').write_text('window.FIVE_SITE_REPORT = '+json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')+';\n',encoding='utf-8',newline='\n')
    summary_rows=[]; selection_rows=[]; longest_rows=[]
    for g in valid:
        d=g['distribution']['days_ge_threshold']; selection=g['selection']
        summary_rows.append([LABELS[g['grid_id']],g['grid_id'],g['complete_summers'],fmt(g['mean_days_ge_threshold_per_complete_summer'],2),*[fmt(d[k]) for k in ['min','p25','median','p75','max']],len(g['episodes'])])
        for role,label in [('typical','典型夏季'),('extreme','阈值天数最多夏季')]:
            choice=selection[role]; year=choice['year']; a=next(a for a in g['annual'] if a['year']==year)
            candidates=choice.get('candidate_years',choice.get('candidate_years_max_days'))
            selection_rows.append([LABELS[g['grid_id']],label,year,', '.join(map(str,candidates)),a['days_ge_threshold'],a['max_consecutive_days'],a['episode_count']])
        maximum=max(e['duration_days'] for e in g['episodes'])
        for e in g['episodes']:
            if e['duration_days']==maximum:longest_rows.append([LABELS[g['grid_id']],e['start_date'],e['end_date'],e['duration_days'],fmt(e['peak_hi_c'],2),e['peak_date'],'有' if e['left_season_censored'] or e['right_season_censored'] else '无'])
    site_by_id={s['site_id']:s for s in catalog['sites']}; grid_by_id={g['grid_id']:g for g in temporal['grids']}
    audit_by_id={a['site_id']:a for a in spatial['site_audits']}
    cases=[]
    for link in temporal['site_links']:
        s=site_by_id[link['site_id']]; g=grid_by_id[link['primary_grid_id']]; a=audit_by_id[link['site_id']]
        climate='原主网格未通过项目陆地掩膜，年度与连续过程均为空；北侧有效格只保留为空间情景。' if not g['annual'] else f'主网格 {g["grid_id"]}；25个完整夏季。平均阈值天数{g["mean_days_ge_threshold_per_complete_summer"]:.2f}天/夏季，典型年{g["selection"]["typical"]["year"]}，阈值天数最多年{g["selection"]["extreme"]["year"]}。'+('与另外两个杭州场所共享全部气候序列。' if len(g['site_ids'])==3 else '')
        special='此前包围盒中心位于凹多边形外，因此采用内部区段中点；本轮计算的真实面积质心在厂内，两者不是同一种点。' if s['site_id']=='factory_002' else '四类空间情景均未改变主匹配网格。'
        cases.append(f'<article class="case"><h3>{html.escape(s["name"])}</h3><p>{html.escape(s["entity"])}<br>生产地址：{html.escape(s["production_address"])}</p><p>{climate}</p><p class="small">地图多边形面积约{a["area_m2"]/1e6:.4f} km²。{special}</p><details><summary>身份与地址核验记录</summary><p class="small">{html.escape(s["verification_note"])}<br>{html.escape(s["evidence_date_note"])}<br>{html.escape(s["evidence_locators"])}</p></details></article>')
    spatial_rows=[[a['name'],fmt(a['area_m2'],2),len(a['positive_area_grid_ids']),fmt(a['primary_grid_area_fraction']*100,2)+'%',fmt(a['land_mask_accepted_area_fraction']*100,2)+'%',f'{a["changed_grid_scenario_count"]} / {a["scenario_count"]}'] for a in spatial['site_audits']]
    scenario_rows=[[r['name'],SCENARIOS[r['scenario']],('—' if r['longitude'] is None else f'{r["latitude"]:.6f}, {r["longitude"]:.6f}'),('不适用' if r['inside_boundary'] is None else '是' if r['inside_boundary'] else '否'),r['grid_id'],int(r['land_mask']),fmt(r['mean_days_ge_threshold_per_complete_summer'],2),fmt(r['mean_days_difference_from_primary'],2)] for r in spatial['representative_scenarios']]
    tesla=audit_by_id['factory_001']; censor_count=sum(e['left_season_censored'] or e['right_season_censored'] for e in episodes)
    stamp=datetime.now(timezone.utc).isoformat()
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五场所研究报告 · EHHR</title><link rel="stylesheet" href="report.css"><script src="report-data.js" defer></script><script src="report.js" defer></script></head><body>
<header><div class="meta">EHHR-FIVE-SITE-01 · 阶段1—3 · 研究报告</div><h1>首批五场所：历史湿热环境与空间匹配审查</h1><p class="meta">气候时期2000—2024年JJA · 证据快照2026-10-29 · 方案规则固定2026-11-05<br>报告生成 {stamp} · <a href="../index.html">返回研究工作台</a></p></header><main>
<nav aria-label="报告目录"><a href="#findings">主要结果</a><a href="#protocol">研究口径</a><a href="#temporal">年度与过程</a><a href="#cases">五场所案例</a><a href="#spatial">空间审查</a><a href="#methods">方法核查</a><a href="#episodes">完整过程表</a><a href="#downloads">资料下载</a></nav>
<section id="findings"><h2>1. 本轮主要结果</h2><div class="lead"><p><strong>5个已核验场所，只有2个有效主气候网格。</strong>四个有效场所对应50个完整网格年；杭州3厂共享同一序列。特斯拉主格未通过陆地掩膜，保持空值。</p><p>两格共识别<strong>{len(episodes)}段阈值连续过程</strong>。两格的阈值天数最多夏季都是2022年；采埃孚对应格的全期最长连续过程则发生在2017年。天数最多与持续最久应分别判读。</p><p>特斯拉存档多边形的<strong>{tesla['primary_grid_area_fraction']*100:.2f}%</strong>落在原主格分区，其余<strong>{tesla['land_mask_accepted_area_fraction']*100:.2f}%</strong>落在北侧有效分区。四类空间情景均未改变主格；不能用北侧格补齐全厂结果。</p></div><p class="caption">本报告分析当前映射厂址的历史网格环境。网格掩膜颜色与面积比例不表示厂区真实海陆性质；本轮没有新增现场或在营核验。</p></section>
<section id="protocol"><h2>2. 研究口径与固定选择规则</h2><p>纳入首批5个公开资料已核验场所，排除30个待核验候选。分析单元按主网格去重，不把同格企业重复计数，也不假定不同格、不同年份或连续过程相互独立。内部方案在新增分析前固定，既有汇总已知，因此不称为事前注册。</p><p>使用北京时间日最高HI，以<strong>≥40.6°C</strong>为项目固定操作阈值。每个JJA的92天必须完整且有限；缺日期或缺值不填0。连续过程保留单日段、跨季重新开始，不能跨缺测连接。</p>{table(['对象','固定规则','并列处理'],[['典型夏季','年度阈值天数最接近完整夏季中位数','取最早年'],['阈值天数最多夏季','年度阈值天数最大','先最长连续更长，再取最早年'],['年际分布','min / P25 / median / P75 / max','线性分位方法；不是置信区间'],['季节边界','6月1日开始或8月31日结束的过程标记截断','时长只表示JJA内部分']])}<p class="small"><a href="downloads/ESG_FIVE_SITE_PROTOCOL.md" download>完整研究方案</a> · <a href="downloads/protocol.json" download>机器可读规则</a> · <a href="downloads/ESG_FIVE_SITE_DATA_DICTIONARY.md" download>数据字典</a></p></section>
<section id="temporal"><h2>3. 年度分布、典型夏季与连续过程</h2><h3>表1 · 按网格去重的年度阈值天数</h3>{table(['关联','网格（°N, °E）','完整夏季','均值/天','最少/天','P25/天','中位/天','P75/天','最多/天','过程数'],summary_rows)}{figure_html('annual-comparison','图1 · 各年JJA的阈值天数D与最长连续天数L，统一0—92天坐标；未拟合趋势。')}<h3>表2 · 按固定规则选择的夏季</h3>{table(['关联','选择用途','选定年','第一规则并列候选','D/天','L/天','过程数'],selection_rows)}{figure_html('selected-summers','图2 · 按固定规则选出的典型年与阈值天数最多年。统一HI纵轴，虚线为40.6°C；浅色背景表示达到阈值的日期。')}<h3>表3 · 全期最长连续过程（保留全部并列）</h3>{table(['关联','开始','结束','持续/天','段内峰值HI/°C','峰值日期','季节截断'],longest_rows)}<p>全期共有{censor_count}段过程触及JJA边界。这些过程在季节外是否延续未检查，不能把JJA内时长当作完整气象过程时长。407段不是407个相互独立的天气事件。</p></section>
<section id="cases"><h2>4. 五个场所的关联案例</h2>{''.join(cases)}</section>
<section id="spatial"><h2>5. 真实地图多边形与空间情景</h2><p>本轮对存档OSM多边形与最近中心分区做实际相交，替代此前仅依据包围盒的候选判断。面积采用每厂本地LAEA等积投影，经纬线段加密后计算，并用半步精化和WGS84测地面积交叉核对。空间分区是项目匹配规则，不是厂区微环境范围。</p>{table(['场所','多边形面积/m²','正面积网格数','原主格覆盖','掩膜接受格覆盖','改变主格情景数'],spatial_rows)}{figure_html('polygon-grid-audit','图3 · 五个存档厂区的实际交集与代表点。各子图范围不同；灰色只表示对应网格未通过项目掩膜，不是厂区真实地表分类。')}<p>5厂共6个正面积交集。面积质心、多边形内部点及最大覆盖格与原主代表点均选中相同主格；这一结果只表明本轮四类情景未改变选择，不证明位置没有误差。福特的真实面积质心在厂内，与此前位于厂外的包围盒中心应区分。</p><details><summary>表4 · 展开全部20种定位情景</summary>{table(['场所','情景','纬度,经度','点在厂内','所选格','掩膜','平均阈值天数','相对主结果差/天'],scenario_rows)}</details><p class="note">特斯拉主格占约70.05%，北侧有效格约29.95%。北侧格的10.52天/夏季与14天最长连续值仍只是附加情景。主格缺值使数值差不可计算；不剔除掩膜拒绝部分，也不把有效部分重新归一化为全厂估计。</p></section>
<section id="methods"><h2>6. 方法核查与解释边界</h2><p>已有处理链从ERA5温度与露点推导相对湿度，再计算小时HI并取北京时间日最高值。一个有效日要求24个整点小时；有小时NaN则日值为NaN。本轮检查了当前代码与相关测试，没有重新计算25个夏季的小时输入链。</p><p>HI变量保留了原气温的GRIB名称和K单位标签；实际派生单位是Celsius。本轮保留原属性并按派生定义读取，不再次减273.15。0.25°是CDS提供的规则经纬度再网格产品；<code>lsm≥0.5</code>是项目筛选规则。</p><p>NWS不同官方材料中存在105°F与103°F起始的两套分级示例。40.6°C接近105°F的一位小数近似，沿用作本项目操作阈值，不能称为统一官方分级或职业限值。HI也不能替代WBGT、厂内测量或员工暴露评价。</p><ul class="references"><li><a href="https://www.weather.gov/tbw/heatindex">NWS · 热指数公式和适用范围</a></li><li><a href="https://www.weather.gov/tsa/heatindex">NWS Tulsa · 105°F口径</a>；<a href="https://www.weather.gov/ama/heatindex">NWS Amarillo · 103°F口径</a></li><li><a href="https://www.weather.gov/safety/heat-index">NWS · HI与WBGT工具区别</a></li><li><a href="https://confluence.ecmwf.int/pages/viewpage.action?pageId=76414402">ECMWF · ERA5资料说明</a>；<a href="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview">CDS · ERA5规则网格产品</a></li><li><a href="https://proj.org/en/stable/operations/projections/laea.html">PROJ · LAEA等积投影</a>；<a href="https://shapely.readthedocs.io/en/2.1.2/reference/shapely.intersection.html">Shapely · 几何相交</a></li></ul><p class="small"><a href="downloads/ESG_FIVE_SITE_METHOD_AUDIT.md" download>下载方法核查记录（源码行号、官方来源与验证范围）</a></p></section>
<section id="episodes"><h2>7. 完整阈值连续过程表</h2><p class="caption">表内HI显示两位小数，导出保留原精度。每行是一个网格的连续日期段，同一天不同格可对应同一天气过程。</p><div class="filters"><label>网格<select id="episode-grid"><option value="all">全部有效主格</option></select></label><label>年份<select id="episode-year"><option value="all">全部年份</option></select></label><label>最少持续天数<input id="episode-min" type="number" min="1" max="92" value="1"></label><button id="episode-export">导出当前过程 CSV</button></div><p id="episode-count" class="small" aria-live="polite"></p><div id="episode-table" class="table-wrap"><table><thead><tr><th>网格</th><th>开始</th><th>结束</th><th>持续/天</th><th>峰值HI/°C</th><th>峰值日期</th><th>段内均值HI/°C</th><th>季节边界截断</th></tr></thead><tbody id="episode-rows"></tbody></table></div></section>
<section id="downloads"><h2>8. 复现资料与数据下载</h2><p>年度与过程模块逐项核对原研究包；空间模块检查面积闭合、掩膜保留、步长精化与测地面积。各模块清单保留协议、输入和输出SHA-256及依赖版本。上述核对属于作者自审，不是现场核验或对原始气候文件的独立再现。</p><div class="downloads"><a href="five-site-study.zip" download>完整研究资料 ZIP</a><a href="downloads/temporal/annual.csv" download>50个网格年度记录</a><a href="downloads/temporal/episodes.csv" download>407段连续过程</a><a href="downloads/temporal/year_selection.csv" download>选年规则与并列记录</a><a href="downloads/spatial/site_spatial_audit.csv" download>场所空间审查</a><a href="downloads/spatial/polygon_grid_intersections.csv" download>多边形面积交集</a><a href="downloads/spatial/representative_scenarios.csv" download>20种定位情景</a><a href="downloads/spatial/factory_grid_intersections.geojson" download>实际交集GeoJSON</a><a href="downloads/temporal/manifest.json" download>时间分析清单</a><a href="downloads/spatial/spatial_manifest.json" download>空间分析清单</a><a href="report-manifest.json" download>报告产物清单</a></div><p class="small">资料包含分析脚本，需在原科学项目与现有输入数据环境中复现；不包含原始NetCDF或第三方来源全文。研究规则见方案与数据字典。</p></section>
</main><footer>资料快照2026-10-29 · 本轮只完成既定第1—3步，未扩样。地图数据 © OpenStreetMap contributors · ODbL；其他材料保留原权利。<br><a href="../index.html">返回研究工作台</a></footer></body></html>'''
    (output/'index.html').write_text(report,encoding='utf-8',newline='\n')
    with zipfile.ZipFile(output/'five-site-study.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['downloads','figures']:
            for file in sorted((output/directory).rglob('*')):
                if file.is_file():archive.write(file,str(file.relative_to(output)))
    inputs={'protocol':sha(protocol_path),'temporal_manifest':sha(temporal_dir/'manifest.json'),'spatial_manifest':sha(spatial_dir/'spatial_manifest.json'),'research_data':sha(research_path),'catalog':sha(catalog_path)}
    manifest={'generated_at_utc':stamp,'protocol_id':protocol['protocol_id'],'review_type':'Author self-review, not field or independent verification','input_sha256':inputs,'source_code_sha256':sha(Path(__file__)),'output_sha256':{str(p.relative_to(output)).replace('\\','/'):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='report-manifest.json'}}
    (output/'report-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return {'verified_sites':len(temporal['site_links']),'valid_primary_grids':len(valid),'grid_years':sum(len(g['annual']) for g in valid),'episodes':len(episodes),'scenario_rows':len(spatial['representative_scenarios']),'output_files':len(manifest['output_sha256'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['temporal','spatial','research','catalog','protocol','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.temporal,args.spatial,args.research,args.catalog,args.protocol,args.output)))
