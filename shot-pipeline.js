const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const base = 'file://' + path.resolve(__dirname, 'index.html');

  page.on('console', m => { if (m.type() === 'error') console.log('[console.error]', m.text()); });
  page.on('pageerror', e => console.log('[pageerror]', e.message));

  await page.goto(base);
  await page.waitForFunction('window.__ready === true');
  await page.waitForTimeout(1500);

  // 各阶段截图：跳到阶段后等动画演到中段
  const shots = [
    { idx: 0, wait: 2500, name: 'overview' },
    { idx: 2, wait: 3200, name: 'tokenize' },
    { idx: 3, wait: 3200, name: 'embedding' },
    { idx: 4, wait: 2600, name: 'layer-attn' },   // p≈0.43 注意力连线
    { idx: -1, wait: 2400, name: 'layer-moe' },   // 不重置，继续到 p≈0.83 专家点亮
    { idx: 9, wait: 4000, name: 'output-softmax' },
    { idx: 9, wait: 6300, name: 'output-sampled' } // 重新进入，等到采样揭晓 p≈0.9
  ];
  for (const s of shots) {
    if (s.idx >= 0) await page.evaluate(i => window.PIPELINE.debugGo(i), s.idx);
    await page.waitForTimeout(s.wait);
    const out = `/tmp/k3-pipe-${s.name}.png`;
    await page.screenshot({ path: out });
    console.log('saved', out);
  }

  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
