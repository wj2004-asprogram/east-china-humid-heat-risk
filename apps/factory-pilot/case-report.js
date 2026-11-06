'use strict';
const study=window.FIVE_SITE_REPORT;
const byId=id=>document.getElementById(id);
for(const grid of study.grids)byId('episode-grid').append(new Option(grid.label,grid.grid_id));
for(const year of [...new Set(study.episodes.map(r=>r.year))].sort())byId('episode-year').append(new Option(year,year));
let filtered=[];
function renderEpisodes(){
 const minimum=Math.max(1,Number(byId('episode-min').value)||1);
 filtered=study.episodes.filter(r=>(byId('episode-grid').value==='all'||r.grid_id===byId('episode-grid').value)&&(byId('episode-year').value==='all'||r.year===Number(byId('episode-year').value))&&r.duration_days>=minimum);
 byId('episode-count').textContent=`当前 ${filtered.length} / ${study.episodes.length} 段；按网格、日期排列，同格场所不重复计数。`;
 byId('episode-export').disabled=filtered.length===0;
 byId('episode-rows').replaceChildren(...filtered.map(r=>{
  const tr=document.createElement('tr');
  const edge=[r.left_season_censored?'6月1日开始':'',r.right_season_censored?'8月31日结束':''].filter(Boolean).join('；')||'无';
  for(const v of [r.grid_id,r.start_date,r.end_date,r.duration_days,r.peak_hi_c.toFixed(2),r.peak_date,r.mean_hi_c.toFixed(2),edge]){const td=document.createElement('td');td.textContent=v;tr.append(td);}
  return tr;
 }));
}
for(const id of ['episode-grid','episode-year'])byId(id).addEventListener('change',renderEpisodes);
byId('episode-min').addEventListener('input',renderEpisodes);
byId('episode-export').addEventListener('click',()=>{
 const keys=['episode_id','grid_id','threshold_c','year','start_date','end_date','duration_days','peak_hi_c','peak_date','mean_hi_c','left_season_censored','right_season_censored'];
 const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
 const csv='\uFEFF'+[keys,...filtered.map(r=>keys.map(k=>r[k]))].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='five_site_threshold_episodes_filtered.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
renderEpisodes();
