// NODE_PATH may point to an existing Playwright installation.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const output = path.resolve(process.argv[2] || 'output/factory-research-ui-qa');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, deviceScaleFactor: 1 });
  const errors = [], external = [], checks = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('request', r => { if (/^https?:/.test(r.url())) external.push(r.url()); });
  const count = selector => page.locator(selector).count();
  const checkpoint = name => checks.push(name);
  const screenshot = name => page.screenshot({ path: path.join(output, name+'.png'), fullPage: true });
  async function download(selector, filename) {
    const downloading = page.waitForEvent('download');
    await page.locator(selector).click();
    await (await downloading).saveAs(path.join(output, filename));
    return fs.readFileSync(path.join(output, filename), 'utf8');
  }
  // Numeric exports have quoted fields with no embedded newlines.
  const parseNumericCsv = text => text.replace(/^\uFEFF/,'').trim().split(/\r?\n/).map(line => line.slice(1,-1).split('","'));
  try {
    await page.goto(pathToFileURL(path.join(__dirname, 'dist', 'index.html')).href);
    assert.equal(await count('#site-rows tr'), 5);
    assert.equal(await page.inputValue('#active-site'), 'factory_002');
    assert.equal(await count('#grid-rows tr'), 3);
    assert.match(await page.locator('#list-count').innerText(), /2 个不同有效主网格/);
    assert.equal(await page.locator('#study-summary strong').allTextContents().then(v=>v.join(',')), '35,4,100,850');
    checkpoint('verified_default_and_grid_deduplication');
    assert.equal(await count('#geometry polygon'), 1);
    assert.match(await page.locator('#coordinate-audit').innerText(), /最长内部区段/);
    assert.match(await page.locator('#grid-rows').innerText(), /44.12/);
    await screenshot('desktop-sample');
    checkpoint('actual_polygon_and_concave_representative_point');

    await page.click('#tab-time');
    assert.equal(await count('#annual-table tbody tr'), 25);
    assert.equal(await count('#annual-comparison polyline'), 2);
    assert.match(await page.locator('#annual-comparison svg').getAttribute('aria-label'), /0至92/);
    assert.equal(await count('#daily-table tbody tr'), 92);
    await page.selectOption('#daily-year', '2000');
    assert.equal(await count('#daily-table .exceed'), 30);
    assert.match(await page.locator('#annual-table tbody tr:first-child').innerText(), /2000\s+92\s+92\s+0\s+0\s+是\s+30\s+8/);
    assert.equal(await page.locator('#daily-table tbody tr:first-child td:first-child').innerText(), '2000-06-01');
    assert.equal(await page.locator('#daily-table tbody tr:last-child td:first-child').innerText(), '2000-08-31');
    checkpoint('annual_completeness_and_single_summer_daily_values');
    const annual = parseNumericCsv(await download('#export-annual', 'selected-annual.csv'));
    assert.equal(annual.length, 26);
    assert.equal(annual[1][0], '30.2500,120.5000');
    assert.equal(annual[1].at(-2), '30');
    assert.equal(annual[1].at(-1), '8');
    const daily = parseNumericCsv(await download('#export-daily', 'selected-daily.csv'));
    assert.equal(daily.length, 93);
    const raw = await page.evaluate(() => window.FACTORY_RESEARCH.grids.find(g=>g.grid_id==='30.2500,120.5000').daily.filter(d=>d.date.startsWith('2000-')));
    assert.deepEqual(daily.slice(1).map(row=>Number(row[2])), raw.map(d=>d.daily_max_heat_index_c));
    assert.equal(daily.slice(1).filter(row=>row[4]==='true').length, 30);
    checkpoint('annual_and_unrounded_daily_csv');
    await screenshot('desktop-time');

    await page.selectOption('#active-site', 'factory_001');
    assert.equal(await count('#annual-table tbody tr'), 0);
    assert.equal(await count('#daily-table tbody tr'), 0);
    assert.equal(await page.locator('#export-annual').isDisabled(), true);
    assert.equal(await page.locator('#export-daily').isDisabled(), true);
    assert.match(await page.locator('#time-context').innerText(), /当前场所没有主结果/);
    checkpoint('masked_primary_withholds_all_climate_values');
    await page.click('#tab-sensitivity');
    assert.equal(await count('#sensitivity-table tbody tr'), 2);
    const scenarios = await page.locator('#sensitivity-table tbody tr').allTextContents();
    assert.match(scenarios[0], /主匹配/);
    assert.match(scenarios[0], /未通过陆地掩膜/);
    assert.match(scenarios[1], /包围盒候选情景/);
    assert.match(scenarios[1], /10.52/);
    assert.match(scenarios[1], /不可计算/);
    assert.match(await page.locator('#sensitivity-note').innerText(), /主结果仍为空/);
    await screenshot('desktop-sensitivity');
    checkpoint('sensitivity_cannot_replace_primary_or_make_invalid_delta');
    await page.click('#tab-evidence');
    assert.match(await page.locator('#method-table').innerText(), /2026-10-29/);
    assert.match(await page.locator('#method-table').innerText(), /0.5/);
    assert.match(await page.locator('#provenance').innerText(), /GRIB/);
    assert.ok(await count('#sources .source') >= 2);
    checkpoint('evidence_snapshot_mask_rule_and_metadata_disclosure');
    await screenshot('desktop-methods');
    await page.locator('#tab-evidence').focus();
    await page.keyboard.press('Home');
    assert.equal(await page.locator('#tab-sample').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#panel-sample').isVisible(), true);
    checkpoint('keyboard_accessible_tabs');

    await page.selectOption('#verification', 'all');
    assert.equal(await count('#site-rows tr'), 35);
    assert.equal(await count('#grid-rows tr'), 5);
    assert.match(await page.locator('#list-count').innerText(), /4 个不同有效主网格/);
    await page.click('#tab-time');
    assert.equal(await count('#annual-comparison polyline'), 4);
    checkpoint('sensitivity_only_grid_excluded_from_main_comparison');
    await page.click('#tab-sample');
    await page.selectOption('#verification', 'candidate_unverified');
    assert.equal(await count('#site-rows tr'), 30);
    assert.equal(await page.locator('#candidate-notice').isVisible(), true);
    await page.selectOption('#city', '苏州（太仓）');
    assert.equal(await count('#site-rows tr'), 3);
    const filtered = await download('#export', 'filtered-sites.csv');
    assert.equal(filtered.trim().split(/\r?\n/).length, 4);
    assert.equal((filtered.match(/candidate_unverified/g)||[]).length, 3);
    checkpoint('candidate_city_filter_and_filtered_export');
    await page.fill('#search', '<img src=x onerror=alert(1)>');
    assert.equal(await count('#site-rows tr'), 0);
    assert.equal(await count('#site-rows img'), 0);
    assert.equal(await page.locator('#empty').isVisible(), true);
    for (const id of ['export','export-annual','export-daily','active-site','daily-year']) {
      assert.equal(await page.locator('#'+id).isDisabled(), true, id+' should be disabled without a selection');
    }
    assert.equal(await count('#annual-comparison polyline'), 0);
    checkpoint('empty_filter_clears_stale_data_and_disables_exports');
    await page.click('#reset');
    assert.equal(await count('#site-rows tr'), 5);
    await page.selectOption('#sort', 'heat');
    assert.equal(await page.locator('#site-rows tr:last-child .site-link').innerText(), '特斯拉上海超级工厂');
    checkpoint('reset_and_missing_metrics_sort_last');
    await page.selectOption('#active-site', 'factory_003');
    await page.setViewportSize({ width: 390, height: 844 });
    for (const tab of ['sample','time','sensitivity','evidence']) {
      await page.click('#tab-'+tab);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'Mobile overflow in '+tab);
      await page.evaluate(() => window.scrollTo(0,0));
      await screenshot('mobile-'+tab);
    }
    checkpoint('all_mobile_tabs_without_page_overflow');
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    checkpoint('no_javascript_errors_or_external_requests');
    fs.writeFileSync(path.join(output, 'ui_verification.json'), JSON.stringify({
      passed:true, checked_at_utc:new Date().toISOString(), checks,
      browser_version:browser.version(), errors, external_requests:external
    }, null, 2));
    console.log(`UI verification passed: ${checks.length} research/interaction/layout/offline checks`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
