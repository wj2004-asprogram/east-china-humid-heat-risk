/* Verify the local report with an existing headless browser, without a server. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL, fileURLToPath} = require('node:url');
const args = process.argv.slice(2);
const value = key => {const i = args.indexOf(key); return i < 0 ? undefined : args[i + 1];};
const {chromium} = require(value('--playwright') || 'playwright-core');
const root = path.resolve(__dirname, '..');
const output = path.resolve(value('--output') || path.join(root, 'reports/probability_phase5/qa'));

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: value('--browser')});
  const page = await browser.newPage({viewport: {width: 1280, height: 1000}});
  const errors = [], external = [], checks = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('request', r => {if (/^https?:/.test(r.url())) external.push(r.url());});
  try {
    await page.goto(pathToFileURL(path.join(root, 'reports/probability_phase5/index.html')).href);
    assert.equal(await page.locator('article').count(), 5);
    assert.equal(await page.locator('figure img').count(), 4);
    assert.ok(await page.locator('figure img').evaluateAll(images => images.every(i => i.complete && i.naturalWidth > 0)));
    checks.push('five_cards_and_four_loaded_figures');
    const urls = await page.locator('a[href],img[src]').evaluateAll(nodes => nodes.map(n => n.href || n.src));
    for (const url of urls) if (url.startsWith('file:')) assert.ok(fs.existsSync(fileURLToPath(new URL(url))), url);
    checks.push('local_links_and_downloads_resolve');
    for (const width of [1280, 768, 390]) {
      await page.setViewportSize({width, height: 1000});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at ${width}`);
      for (const summary of await page.locator('details summary').all()) await summary.click();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `expanded overflow at ${width}`);
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({path: path.join(output, `report-${width}.png`)});
      for (const summary of await page.locator('details summary').all()) await summary.click();
    }
    checks.push('desktop_tablet_phone_no_page_overflow_with_tables_expanded');
    await page.setViewportSize({width: 1280, height: 1000});
    await page.locator('#factory_002 details summary').click();
    await page.locator('#factory_002').screenshot({path: path.join(output, 'hangzhou-card-expanded.png')});
    assert.match(await page.locator('#factory_001').innerText(), /掩膜拒绝/);
    assert.match(await page.locator('#factory_005').innerText(), /buffer/);
    assert.equal(await page.locator('#factory_001 details').count(), 0);
    checks.push('masked_primary_and_buffer_scope_are_visible');
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    checks.push('no_page_errors_or_external_requests');
    fs.writeFileSync(path.join(output, 'verification.json'), JSON.stringify({
      checked_at_utc: new Date().toISOString(), browser: browser.version(),
      checks, errors, external_requests: external, status: 'passed'
    }, null, 2) + '\n', 'utf8');
    console.log(JSON.stringify({status: 'passed', checks, output}));
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
