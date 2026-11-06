'use strict';

const data = window.FACTORY_PILOT;
const research = window.FACTORY_RESEARCH;
const sites = data.sites;
const grids = new Map(research.grids.map(g => [g.grid_id, g]));
const links = new Map(research.site_grid_links.map(s => [s.site_id, s]));
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const verified = s => s.site_verification === 'verified_public_sources';
const usable = g => g?.screening_status === 'ok';
const complete = a => a.complete_summer === true || a.complete_summer === 'True';
const num = (v, digits = 0) => v === null || v === undefined || v === '' ? '—' : Number(v).toFixed(digits);
const status = g => ({ok:'完整', non_land_grid:'未通过陆地掩膜', invalid_coordinate:'坐标越界 / 无效', incomplete_climate_data:'气候数据不完整'})[g?.screening_status] || '未提供';
const badge = s => `<span class="status${verified(s) ? '' : ' pending'}">${verified(s) ? '公开资料已核验' : '待核验候选'}</span>`;
const safeUrl = value => { try { const u = new URL(value); return ['https:','http:'].includes(u.protocol) ? esc(u.href) : '#'; } catch { return '#'; } };
const cell = (html, cls = '') => `<td${cls ? ` class="${cls}"` : ''}>${html}</td>`;
const numeric = (v, digits = 0) => cell(num(v, digits), 'numeric');
const table = (headers, rows) => `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
const palette = ['#285b85','#a66036','#476c48','#775990','#69757e'];
let active = sites.find(s => verified(s) && usable(grids.get(s.grid_id)))?.site_id;
let visible = [];
let currentTab = 'sample';

const sourceNames = {
  tesla_miit:'工信部 · 特斯拉生产地址', ford_miit:'工信部 · 福特生产地址',
  ford_government:'杭州市政府 · 杭州装配工厂', ferrero_inspection:'张家港市政府 · 食品抽检第74、76行',
  ferrero_locations:'费列罗官网 · 全球布局', gree_ccc:'公开 CCC 证书 · PDF 第1页',
  gree_eia:'杭州环评材料 · PDF 第5、26页', zf_locations:'ZF 官网 · Shanghai Plant 2',
  zf_iso14001:'ZF 公开生产范围 · PDF 第3页', zf_address:'上海市政府 · 中文实体及地址',
  osm_crs:'OSM · WGS84 坐标系说明', minhang_map:'OSM 闵行范围 · 原始响应', taicang_map:'OSM 太仓范围 · 原始响应'
};

function selectedSite() { return sites.find(s => s.site_id === active); }
function selectedGrid() { return grids.get(selectedSite()?.grid_id); }
function visibleGrids() { return [...new Set(visible.map(s => s.grid_id))].sort().map(id => grids.get(id)); }
function coordinateMethod(s) {
  return s.coordinate_method.includes('interior_span')
    ? '中纬线最长内部区段中点；包围盒中心位于边界外'
    : '包围盒中心；已检查位于厂区边界内';
}

function initialize() {
  const v = research.validation;
  $('snapshot').textContent = `证据快照 ${data.reviewed_on}`;
  $('study-summary').innerHTML = [
    [sites.length,`地点；其中${sites.filter(verified).length}个已核验`],
    [v.unique_primary_land_grids,'不同有效主网格'],
    [v.unique_primary_grid_year_records,'不同主网格—年份组合'],
    [v.primary_annual_records_matched,'条场所逐年展开记录（含重复）']
  ].map(([value,label]) => `<span><strong>${value}</strong>${label}</span>`).join('');
  for (const city of [...new Set(sites.map(s => s.city))]) {
    const option = new Option(city, city); $('city').append(option);
  }
  for (let year = Number(research.calendar.first_date.slice(0,4)); year <= Number(research.calendar.last_date.slice(0,4)); year++) {
    $('daily-year').append(new Option(String(year), String(year)));
  }
  $('daily-year').value = research.calendar.last_date.slice(0,4);
  render();
}

function render() {
  const query = $('search').value.trim().toLocaleLowerCase();
  visible = sites.filter(s => ($('verification').value === 'all' || s.site_verification === $('verification').value)
    && ($('city').value === 'all' || s.city === $('city').value)
    && `${s.name} ${s.entity} ${s.production_address} ${s.city}`.toLocaleLowerCase().includes(query));
  visible.sort($('sort').value === 'heat'
    ? (a,b) => (b.mean_days_ge_threshold_per_complete_summer ?? -Infinity) - (a.mean_days_ge_threshold_per_complete_summer ?? -Infinity) || a.site_id.localeCompare(b.site_id)
    : (a,b) => a.site_id.localeCompare(b.site_id));
  if (!visible.some(s => s.site_id === active)) active = visible[0]?.site_id;
  $('list-count').textContent = `当前筛选：${visible.length} 个地点 · ${visible.filter(verified).length} 个已核验 · ${visibleGrids().filter(usable).length} 个不同有效主网格`;
  $('active-site').replaceChildren(...visible.map(s => new Option(`${s.site_id} · ${s.name}`,s.site_id)));
  $('active-site').value = active || '';
  $('active-site').disabled = !visible.length;
  $('export').disabled = !visible.length;
  $('empty').hidden = !!visible.length;
  $('candidate-notice').hidden = !visible.some(s => !verified(s));
  renderSample(); renderTime(); renderSensitivity(); renderEvidence();
}

function renderSample() {
  $('grid-rows').innerHTML = visibleGrids().map(g => {
    const members = visible.filter(s => s.grid_id === g.grid_id);
    return `<tr class="${selectedSite()?.grid_id === g.grid_id ? 'selected' : ''}">${cell(`<button class="grid-link mono" data-site="${esc(members[0].site_id)}">${esc(g.grid_id)}</button>`)}${cell(`${members.length} / ${members.filter(verified).length}`)}${cell(status(g))}${cell(`${num(g.complete_summers)} / ${num(g.total_summers)}`)}${numeric(g.mean_days_ge_threshold_per_complete_summer,2)}${numeric(g.max_consecutive_days)}</tr>`;
  }).join('');
  $('site-rows').innerHTML = visible.map(s => `<tr class="${s.site_id === active ? 'selected' : ''}">${cell(`<button class="site-link" data-site="${esc(s.site_id)}">${esc(s.name)}</button><div class="site-meta">${badge(s)}${esc(s.city)} · ${esc(s.site_id)}</div>`)}${cell(`${num(s.latitude,6)}, ${num(s.longitude,6)}`,'mono')}${cell(esc(s.grid_id),'mono')}${numeric(s.grid_distance_km,3)}${cell(status(grids.get(s.grid_id)))}${cell(`${links.get(s.site_id).bbox_grid_ids.length} 格${links.get(s.site_id).bbox_grid_ids.length > 1 ? '（位置敏感）' : ''}`)}</tr>`).join('');
  const s = selectedSite();
  if (!s) {
    $('detail').innerHTML = '<p>无选中场所。</p>'; $('geometry').innerHTML = ''; $('coordinate-audit').innerHTML = ''; return;
  }
  $('detail').innerHTML = `<h2>${esc(s.name)} ${badge(s)}</h2><p>${esc(s.entity || '企业实体尚未交叉核验')}<br>生产地址：${esc(s.production_address || '待核验')}</p><p class="small muted">${esc(s.verification_note)}</p>`;
  const g = selectedGrid();
  $('coordinate-audit').innerHTML = `<h2>空间匹配记录</h2><dl><dt>坐标系 / 方法</dt><dd>WGS84 · ${coordinateMethod(s)}</dd><dt>主网格中心</dt><dd class="mono">${esc(s.grid_id)} · ${status(g)}</dd><dt>代表点至网格中心的大圆距离</dt><dd>${num(s.grid_distance_km,3)} km；不作为定位误差或置信区间</dd><dt>地图对象 / 版本时间</dt><dd><a href="${safeUrl(s.source_url)}" target="_blank" rel="noopener noreferrer">OSM way ${esc(s.osm_id)}</a> · v${esc(s.osm_version)}<br>${esc(s.osm_timestamp)}</dd><dt>包围盒（西、南、东、北）</dt><dd class="mono">${s.bbox.map(v => num(v,6)).join(', ')}</dd></dl><p class="small muted">地图边界、代表点和网格均为环境筛查定位依据，未测量厂区内工作位置。身份核验与气候数据可用性是两个独立字段。</p>`;
  renderGeometry(s);
}

function renderGeometry(site) {
  const feature = research.boundaries.features.find(f => f.properties.site_id === site.site_id);
  if (!feature) { $('geometry').textContent = '未提供边界。'; return; }
  const ring = feature.geometry.coordinates[0];
  const [west,south,east,north] = site.bbox;
  const lon0 = (west+east)/2, lat0 = (south+north)/2, c = Math.cos(lat0*Math.PI/180);
  const w = 660, h = 310, px = 65, py = 35;
  const extentX = Math.max((east-west)*c*1.4,.004), extentY = Math.max((north-south)*1.4,.004);
  const scale = Math.min((w-px*2)/extentX,(h-py*2)/extentY);
  const x = lon => w/2+(lon-lon0)*c*scale;
  const y = lat => h/2-(lat-lat0)*scale;
  const worldLeft = lon0-(w/2-px)/(c*scale), worldRight = lon0+(w/2-px)/(c*scale);
  const worldBottom = lat0-(h/2-py)/scale, worldTop = lat0+(h/2-py)/scale;
  const cells = links.get(site.site_id).bbox_grid_ids.map(id => {
    const g = grids.get(id);
    return `<rect x="${x(g.longitude-.125)}" y="${y(g.latitude+.125)}" width="${.25*c*scale}" height="${.25*scale}" fill="${usable(g)?'#f0f5f8':'#e4e6e9'}" stroke="#7e8e9c" stroke-width="1" stroke-dasharray="5 3"><title>${esc(id)} · ${status(g)}</title></rect>`;
  }).join('');
  const ticks = [0,.5,1].map(t => {
    const lon = worldLeft+t*(worldRight-worldLeft), lat = worldBottom+t*(worldTop-worldBottom);
    return `<text x="${x(lon)}" y="${h-12}" text-anchor="middle" font-size="10" fill="#586672">${num(lon,4)}°E</text><text x="${px-7}" y="${y(lat)+3}" text-anchor="end" font-size="10" fill="#586672">${num(lat,4)}°N</text>`;
  }).join('');
  $('geometry').innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(site.name)}真实OSM厂区多边形、代表点和最近网格分区"><defs><clipPath id="geometry-clip"><rect x="${px}" y="${py}" width="${w-px*2}" height="${h-py*2}"/></clipPath></defs><g clip-path="url(#geometry-clip)">${cells}<polygon points="${ring.map(([lon,lat]) => `${x(lon)},${y(lat)}`).join(' ')}" fill="#d8e5ec" fill-opacity=".6" stroke="#224e71" stroke-width="2"/><circle cx="${x(site.longitude)}" cy="${y(site.latitude)}" r="4" fill="#9f442b" stroke="white" stroke-width="1.5"/></g><rect x="${px}" y="${py}" width="${w-px*2}" height="${h-py*2}" fill="none" stroke="#a8b4bd"/>${ticks}<text x="${px}" y="18" font-size="10" fill="#586672">蓝线：厂区边界　红点：主代表点　灰底：网格未通过陆地掩膜</text></svg>`;
}

function annualComparison() {
  const available = visibleGrids().filter(g => g.annual.length && g.land_mask === 1);
  if (!available.length) return '<p class="note">当前筛选没有可展示的陆地网格序列。</p>';
  const w=1000,h=250,left=48,right=16,top=16,bottom=30;
  const first=Number(research.calendar.first_date.slice(0,4)),last=Number(research.calendar.last_date.slice(0,4));
  const x=year => left+(year-first)/(last-first)*(w-left-right);
  const y=value => h-bottom-value/92*(h-top-bottom);
  const axes=[0,23,46,69,92].map(v => `<line x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}" stroke="#e0e5e9"/><text x="${left-8}" y="${y(v)+3}" font-size="10" text-anchor="end">${v}</text>`).join('');
  const xticks=[first,first+5,first+10,first+15,first+20,last].map(v => `<text x="${x(v)}" y="${h-8}" text-anchor="middle" font-size="10">${v}</text>`).join('');
  const lines=available.map((g,i) => {
    let segments=[],segment=[];
    for (const a of g.annual) {
      if (complete(a) && a.days_ge_threshold !== null) segment.push(`${x(a.year)},${y(a.days_ge_threshold)}`);
      else { if(segment.length) segments.push(segment); segment=[]; }
    }
    if(segment.length) segments.push(segment);
    return segments.map(points => `<polyline points="${points.join(' ')}" fill="none" stroke="${palette[i%palette.length]}" stroke-width="${selectedSite()?.grid_id===g.grid_id?2.8:1.5}"/>`).join('');
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="按不同网格去重的2000至2024年度阈值天数，统一0至92天纵轴">${axes}${xticks}${lines}<text x="4" y="12" font-size="10">天</text></svg><div class="chart-legend">${available.map((g,i) => `<span><span class="legend-key" style="background:${palette[i%palette.length]}"></span>${esc(g.grid_id)}（当前关联 ${visible.filter(s => s.grid_id===g.grid_id).length} 处）</span>`).join('')}</div>`;
}

function renderTime() {
  const s = selectedSite(), g = selectedGrid();
  $('annual-comparison').innerHTML = annualComparison();
  if (!s) { $('time-context').textContent='无选中场所。'; $('annual-table').innerHTML=''; $('export-annual').disabled=true; renderDaily(); return; }
  $('time-context').innerHTML = `<h2>${esc(s.name)} · 主匹配 ${esc(s.grid_id)}</h2><p>${status(g)}；阈值 HI ≥${research.threshold_c}°C；年度统计限定 JJA。${g.land_mask !== 1 ? '<strong>当前场所没有主结果。图2若有曲线，表示筛选中的其他有效网格。</strong>' : '该网格同时关联的场所共享这条气候序列。'}</p>`;
  $('export-annual').disabled = !g.annual.length;
  $('annual-table').innerHTML = g.annual.length ? table(['年份','预期天数','有效天数','缺日期','非有限值','完整夏季','阈值天数 D','最长连续 L'],g.annual.map(a => `<tr>${numeric(a.year)}${numeric(a.expected_days)}${numeric(a.valid_days)}${numeric(a.missing_date_days)}${numeric(a.nonfinite_days)}${cell(complete(a)?'是':'否')}${numeric(a.days_ge_threshold)}${numeric(a.max_consecutive_days)}</tr>`)) : `<p class="note">${status(g)}，没有年度主结果；未把缺失值填成0。</p>`;
  renderDaily();
}

function selectedDaily() { return (selectedGrid()?.daily || []).filter(a => a.date.startsWith($('daily-year').value+'-')); }

function renderDaily() {
  const daily = selectedDaily();
  $('export-daily').disabled = !daily.length;
  $('daily-year').disabled = !selectedGrid()?.daily.length;
  if (!daily.length) { $('daily-chart').innerHTML=''; $('daily-table').innerHTML='<p class="note">当前主网格没有可展示的逐日数据。</p>'; return; }
  const w=1000,h=220,left=48,right=16,top=18,bottom=30;
  const allValues=research.grids.flatMap(g => g.daily.map(d => d.daily_max_heat_index_c)).filter(v => v !== null && Number.isFinite(v));
  const ymin=Math.min(0,Math.floor(Math.min(...allValues)/10)*10),ymax=Math.max(60,Math.ceil(Math.max(...allValues)/10)*10);
  const x=i => left+i/(daily.length-1)*(w-left-right),y=v => h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
  const guides=Array.from({length:(ymax-ymin)/10+1},(_,i)=>ymin+i*10).map(v => `<line x1="${left}" y1="${y(v)}" x2="${w-right}" y2="${y(v)}" stroke="#e0e5e9"/><text x="${left-8}" y="${y(v)+3}" text-anchor="end" font-size="10">${v}</text>`).join('');
  let segments=[],segment=[];
  daily.forEach((a,i) => { if(a.daily_max_heat_index_c !== null) segment.push(`${x(i)},${y(a.daily_max_heat_index_c)}`); else { if(segment.length)segments.push(segment); segment=[]; } });
  if(segment.length)segments.push(segment);
  const curve=segments.map(points=>`<polyline points="${points.join(' ')}" fill="none" stroke="#285b85" stroke-width="1.6"/>`).join('');
  const ticks=daily.map((a,i)=> a.date.endsWith('-01') || i===daily.length-1 ? `<text x="${x(i)}" y="${h-8}" text-anchor="middle" font-size="10">${esc(a.date.slice(5))}</text>`:'').join('');
  $('daily-chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc($('daily-year').value)}年JJA逐日最高热指数与40.6摄氏度阈值">${guides}${curve}<line x1="${left}" y1="${y(research.threshold_c)}" x2="${w-right}" y2="${y(research.threshold_c)}" stroke="#a66036" stroke-dasharray="5 3"/><text x="${w-right}" y="${y(research.threshold_c)-5}" text-anchor="end" fill="#8b4d28" font-size="10">HI ${research.threshold_c}°C</text>${ticks}<text x="4" y="12" font-size="10">°C</text></svg>`;
  $('daily-table').innerHTML=table(['日期（北京时间）','日最高 HI / °C','是否达到 ≥40.6°C','数据状态'],daily.map(a=>`<tr${a.daily_max_heat_index_c!==null && a.daily_max_heat_index_c>=research.threshold_c?' class="exceed"':''}>${cell(esc(a.date),'mono')}${numeric(a.daily_max_heat_index_c,2)}${cell(a.daily_max_heat_index_c===null?'—':a.daily_max_heat_index_c>=research.threshold_c?'是':'否')}${cell(a.daily_max_heat_index_c===null?'缺失 / 非有限值':'有效')}</tr>`));
}

function renderSensitivity() {
  const s=selectedSite(),main=selectedGrid();
  if(!s){$('sensitivity-context').textContent='无选中场所。';$('sensitivity-table').innerHTML='';$('sensitivity-note').textContent='';return;}
  const alternatives=links.get(s.site_id).bbox_grid_ids.map(id=>grids.get(id));
  $('sensitivity-context').innerHTML=`<p><strong>${esc(s.name)}</strong> · 主代表点 ${num(s.latitude,6)}°N, ${num(s.longitude,6)}°E · 包围盒候选 ${alternatives.length} 格</p>`;
  $('sensitivity-table').innerHTML=table(['情景 / 网格','掩膜值 / 状态','完整夏季','平均阈值天数','最长连续天数','均值相对主结果差值'],alternatives.map(g=>{
    const delta=usable(g)&&usable(main)?g.mean_days_ge_threshold_per_complete_summer-main.mean_days_ge_threshold_per_complete_summer:null;
    return `<tr${g.grid_id===main.grid_id?' class="selected"':''}>${cell(`${g.grid_id===main.grid_id?'主匹配':'包围盒候选情景'}<br><span class="mono">${esc(g.grid_id)}</span>`)}${cell(`${num(g.land_mask)} · ${status(g)}`)}${cell(`${num(g.complete_summers)} / ${num(g.total_summers)}`)}${numeric(g.mean_days_ge_threshold_per_complete_summer,2)}${numeric(g.max_consecutive_days)}${cell(delta===null?'不可计算':`${delta>0?'+':''}${num(delta,2)} 天`,'numeric')}</tr>`;
  }));
  $('sensitivity-note').textContent=!usable(main)
    ? '主匹配未通过筛查，不能计算相对主结果的数值差。候选情景中的有效结果仅用于说明位置选择的影响，主结果仍为空。'
    : alternatives.length===1?'按当前包围盒口径没有跨网格情景。这不意味着位置没有误差，也不说明网格能够解析厂区微环境。':'同一厂区的包围盒涉及多个分区；应进一步核对边界和代表点。表中差值仅为情景差异，不作为不确定性的统计区间。';
}

function renderEvidence() {
  const s=selectedSite(),meta=research.source_metadata;
  $('method-table').innerHTML=table(['项目','本次采用'],[
    ['气候时期',`${research.calendar.first_date} — ${research.calendar.last_date}；JJA；UTC+8`],
    ['身份/地址证据快照',data.reviewed_on+'；未因界面修订而更新核验日期'],
    ['研究包生成时间',research.generated_at_utc],
    ['基础变量','daily_max_heat_index_c；0.25°；日最高 HI，Celsius'],
    ['阈值和完整性',`HI ≥${research.threshold_c}°C；每夏季预期92天；缺日期或非有限值使年度不完整`],
    ['空间匹配','各经纬度方向最近中心；等距时沿原文件顺序；不替换非陆地格'],
    ['原文件掩膜规则',meta.land_mask_rule],
    ['情景口径','包围盒可能涉及的最近中心分区，未作为真实多边形覆盖确认']
  ].map(([a,b])=>`<tr>${cell(esc(a))}${cell(esc(b))}</tr>`));
  $('evidence-context').innerHTML=s?`<h3>${esc(s.name)}</h3><p>${esc(s.verification_note)}</p><p class="small"><strong>证据日期：</strong>${esc(s.evidence_date_note)}<br><strong>原文定位：</strong>${esc(s.evidence_locators)}</p><p class="small muted">员工人数、班次、防暑措施均无场所级证据，保持空缺；证书仅作为公开生产地址/范围依据，未实时核验认证状态。</p>`:'<p>无选中场所。</p>';
  $('sources').innerHTML=(s?.evidence_ids||[]).map(id=>{
    const source=data.sources[id];
    return `<article class="source"><a href="${safeUrl(source.requested_url)}" target="_blank" rel="noopener noreferrer">${esc(sourceNames[id]||'OSM · 厂区原始边界')} ↗</a><p>获取时间 ${esc(source.retrieved_at_utc)} · ${num(source.bytes)} bytes</p><p class="hash">SHA-256 ${esc(source.sha256)}</p><details><summary>许可、URL和归档文件</summary><p>${esc(source.license_note)}</p><p class="hash">${esc(source.requested_url)}</p><p>${esc(source.raw_file)}</p></details></article>`;
  }).join('');
  $('provenance').innerHTML=`<p class="small">原850条场所年度展开记录与研究包相符，按主网格去重后为100个网格—年份组合。研究包额外包含仅用于敏感性分析的网格；不计入4个有效主网格。</p><div class="table-wrap">${table(['输入','SHA-256'],Object.entries(research.input_sha256).map(([key,value])=>`<tr>${cell(esc(key))}${cell(esc(value),'hash')}</tr>`))}</div><p class="note"><strong>元数据核查记录：</strong>原日最高热指数文件残留原始气温的 GRIB 名称和 K 单位标签。本研究依据派生变量的 units=Celsius、long_name、description 及热指数处理代码使用摄氏单位，不再次减去273.15。原始属性保留在下方供核查，未修改原 NetCDF 或数值。</p><details><summary>展开原始变量属性和研究包校验记录</summary><pre class="hash">${esc(JSON.stringify({source_metadata:meta,validation:research.validation},null,2))}</pre></details><p class="small">复现入口：<code>tools/esg_build_research_data.py</code>；执行说明见 <a href="../README.md">本地 README</a>。所有数值均读取已有气候输出，未重跑下载或小时气象处理。</p>`;
}

function saveCsv(name, columns, records) {
  const quote=value=>{let t=String(value??'');if(/^[\s]*[=+@-]/.test(t)||/^[\t\r\n]/.test(t))t="'"+t;return '"'+t.replace(/"/g,'""')+'"';};
  const content='\uFEFF'+[columns,...records.map(r=>columns.map(k=>r[k]))].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function changeTab(name, focus=false) {
  currentTab=name;
  document.querySelectorAll('[data-tab]').forEach(button=>{const selected=button.dataset.tab===name;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;$('panel-'+button.dataset.tab).hidden=!selected;if(selected&&focus)button.focus();});
}

for(const id of ['verification','city','sort'])$(id).addEventListener('change',render);
$('search').addEventListener('input',render);
$('active-site').addEventListener('change',()=>{active=$('active-site').value;render();});
$('daily-year').addEventListener('change',renderDaily);
$('reset').addEventListener('click',()=>{$('search').value='';$('verification').value='verified_public_sources';$('city').value='all';$('sort').value='id';render();});
$('export').addEventListener('click',()=>saveCsv('factory_screening_filtered.csv',['site_id','name','site_verification','city','production_address','latitude','longitude','coordinate_crs','coordinate_method','screening_status','grid_id','grid_distance_km','complete_summers','total_summers','mean_days_ge_threshold_per_complete_summer','max_consecutive_days','source_url'],visible));
$('export-annual').addEventListener('click',()=>{const g=selectedGrid();if(g?.annual.length)saveCsv('selected_grid_annual.csv',['grid_id','year','expected_days','valid_days','missing_date_days','nonfinite_days','complete_summer','threshold_c','days_ge_threshold','max_consecutive_days'],g.annual.map(a=>({grid_id:g.grid_id,threshold_c:research.threshold_c,...a})));});
$('export-daily').addEventListener('click',()=>saveCsv('selected_grid_daily_'+$('daily-year').value+'.csv',['grid_id','date','daily_max_heat_index_c','threshold_c','ge_threshold'],selectedDaily().map(a=>({grid_id:selectedGrid().grid_id,...a,threshold_c:research.threshold_c,ge_threshold:a.daily_max_heat_index_c===null?'':a.daily_max_heat_index_c>=research.threshold_c}))));
document.addEventListener('click',event=>{const site=event.target.closest('[data-site]'),tab=event.target.closest('[data-tab]');if(site){active=site.dataset.site;render();}if(tab)changeTab(tab.dataset.tab);});
document.querySelector('.tabs').addEventListener('keydown',event=>{const names=['sample','time','sensitivity','evidence'];if(['ArrowRight','ArrowLeft','Home','End'].includes(event.key)){event.preventDefault();let i=names.indexOf(currentTab);i=event.key==='Home'?0:event.key==='End'?names.length-1:(i+(event.key==='ArrowRight'?1:-1)+names.length)%names.length;changeTab(names[i],true);}});
initialize();changeTab('sample');
