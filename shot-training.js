const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const base = 'file://' + path.resolve(__dirname, 'test-training.html');

  page.on('console', m => { if (m.type() === 'error') console.log('[console.error]', m.text()); });
  page.on('pageerror', e => console.log('[pageerror]', e.message));

  // 接口自检
  await page.goto(base + '#pretrain');
  await page.waitForFunction('window.__ready === true');
  const iface = await page.evaluate(() => {
    const m = window.MODULE_TRAINING;
    return ['init','show','hide','update','dispose','screenshot']
      .map(k => k + ':' + (typeof m[k])).join(' ');
  });
  console.log('interface:', iface);

  // 三阶段截图（等动画跑出粒子流和权重变化）
  // 注意：同页 hash 跳转不触发 reload，直接调 show() 切换阶段（顺带验证 stage 切换路径）
  for (const stage of ['pretrain', 'sft', 'rl']) {
    await page.evaluate(s => {
      window.MODULE_TRAINING.show({ stage: s });
      document.querySelectorAll('.bar button').forEach(b =>
        b.classList.toggle('active', b.dataset.stage === s));
    }, stage);
    await page.waitForTimeout(7000);
    const out = `/tmp/module-training-${stage}.png`;
    await page.screenshot({ path: out });
    console.log('saved', out);
  }

  // 模块自带 screenshot() 接口验证
  const dataUrl = await page.evaluate(() => window.MODULE_TRAINING.screenshot().slice(0, 40));
  console.log('screenshot() returns:', dataUrl);

  await browser.close();
})();
