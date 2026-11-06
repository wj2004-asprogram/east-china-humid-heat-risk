const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL,fileURLToPath}=require('node:url');
(async()=>{
 const output=path.resolve(process.argv[2]||'output/five-site-report-qa');fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1024,height:1024},hasTouch:true});
 const errors=[],external=[],checks=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url());});
 try{
  await page.goto(pathToFileURL(path.join(__dirname,'dist','index.html')).href);
  await page.locator('a[href="study/index.html"]').click();
  assert.match(await page.title(),/五场所研究报告/);
  assert.equal(await page.locator('#cases .case').count(),5);
  assert.equal(await page.locator('#episode-rows tr').count(),407);
  assert.equal(await page.locator('#spatial details tbody tr').count(),20);
  assert.match(await page.locator('#findings').innerText(),/70.05%/);
  assert.match(await page.locator('#findings').innerText(),/29.95%/);
  checks.push('entry_five_cases_deduplicated_episodes_and_spatial_results');
  const localUrls=await page.locator('a[href],img[src],script[src],link[href]').evaluateAll(nodes=>nodes.map(n=>n.href||n.src).filter(u=>u.startsWith('file:')));
  for(const url of localUrls)assert.ok(fs.existsSync(fileURLToPath(new URL(url))),'Missing '+url);
  assert.equal(await page.locator('figure img').evaluateAll(imgs=>imgs.every(i=>i.complete&&i.naturalWidth>0)),true);
  checks.push('all_local_links_downloads_and_figures_exist');
  await page.selectOption('#episode-grid','31.0000,121.5000');
  await page.selectOption('#episode-year','2017');await page.fill('#episode-min','30');
  assert.equal(await page.locator('#episode-rows tr').count(),1);
  assert.match(await page.locator('#episode-rows').innerText(),/2017-07-04\s+2017-08-08\s+36/);
  const downloading=page.waitForEvent('download');await page.click('#episode-export');
  const downloaded=await downloading;await downloaded.saveAs(path.join(output,'zf-2017-longest.csv'));
  const csv=fs.readFileSync(path.join(output,'zf-2017-longest.csv'),'utf8');assert.equal(csv.trim().split(/\r?\n/).length,2);
  const expected=await page.evaluate(()=>window.FIVE_SITE_REPORT.episodes.find(e=>e.grid_id==='31.0000,121.5000'&&e.year===2017&&e.duration_days===36));
  assert.ok(csv.includes(String(expected.peak_hi_c)));assert.ok(csv.includes(String(expected.mean_hi_c)));
  checks.push('grid_year_duration_filters_and_full_precision_export');
  await page.selectOption('#episode-grid','all');await page.selectOption('#episode-year','2022');
  assert.equal(await page.locator('#episode-rows tr').count(),2);
  assert.match(await page.locator('#episode-rows').innerText(),/2022-07-19\s+2022-08-26\s+39/);
  checks.push('most_threshold_days_year_distinct_from_all_time_longest');
  await page.fill('#episode-min','92');assert.equal(await page.locator('#episode-rows tr').count(),0);
  assert.equal(await page.locator('#episode-export').isDisabled(),true);checks.push('empty_episode_filter_disables_export');
  await page.fill('#episode-min','1');await page.selectOption('#episode-year','all');
  for(const width of [1024,768,390]){
   await page.setViewportSize({width,height:1024});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`overflow ${width}`);
  }
  checks.push('tablet_landscape_portrait_and_phone_without_page_overflow');
  await page.setViewportSize({width:1024,height:1024});
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(output,'report-top.png')});
  await page.locator('#spatial').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(output,'report-spatial.png')});
  await page.locator('#spatial details summary').click();assert.equal(await page.locator('#spatial details table').isVisible(),true);
  checks.push('expandable_twenty_scenario_table');
  await page.locator('header a').click();assert.equal(await page.locator('#site-rows tr').count(),5);
  checks.push('return_to_existing_workbench');
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);checks.push('no_javascript_errors_or_external_requests');
  fs.writeFileSync(path.join(output,'study_ui_verification.json'),JSON.stringify({passed:true,checked_at_utc:new Date().toISOString(),checks,browser_version:browser.version(),errors,external_requests:external},null,2));
  console.log(`Study report: ${checks.length} checks passed`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
