# -*- coding: utf-8 -*-
"""把 demo_game.json 渲染成一个可以逐步回放的棋盘页面。

用法：
    python render_game.py

输出：demo_game.html（单文件，数据内嵌）
"""

from __future__ import print_function

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_key_steps(moves, side):
    """挑出 5 个有信息量的手，而不是随便截几张图。"""
    if not moves:
        return []
    chosen = []

    def pos(mv):
        return '(%d,%d)' % (mv % side, mv // side)

    def add(i, reason):
        if i is None:
            return
        if i not in [c[0] for c in chosen]:
            chosen.append([i, reason])

    def delta(m):
        return dict(m['search']).get(m['move'], 0.0) - \
            dict(m['prior']).get(m['move'], 0.0)

    add(0, '开局第一手：没有搜索历史，落点完全由网络先验决定')

    gain = max(moves, key=delta)
    g = delta(gain)
    if g >= 0.05:
        add(gain['i'], '搜索修正先验最多的一手：首选点概率从 %.2f 抬到 %.2f'
            % (dict(gain['prior']).get(gain['move'], 0), gain['confidence']))
    else:
        add(gain['i'], '搜索几乎没能改变先验：首选点概率只从 %.2f 挪到 %.2f'
            % (dict(gain['prior']).get(gain['move'], 0), gain['confidence']))

    least = min(moves, key=lambda m: m['confidence'])
    most = max(moves, key=lambda m: m['confidence'])
    add(least['i'], '最不自信的一手：%d 次模拟只有 %.0f%% 落在首选点'
        % (least['root_visits'], least['confidence'] * 100))
    add(most['i'], '最自信的一手：%.0f%% 的模拟集中在一个点'
        % (most['confidence'] * 100))

    for m in moves:
        if m['prior'] and m['search'] and m['prior'][0][0] != m['search'][0][0]:
            add(m['i'], '搜索否定了网络首选：先验看好 %s，搜索改推 %s'
                % (pos(m['prior'][0][0]), pos(m['search'][0][0])))
            break

    add(len(moves) - 1, '终局：第 %d 手结束' % len(moves))

    # 去重后不足 5 个时，按手数等距补位，保证能看出中盘走势
    if len(chosen) < 5 and len(moves) > 4:
        targets = [int(round(len(moves) * k / 4.0)) - 1 for k in (1, 2, 3)]
        for t in targets:
            if len(chosen) >= 5:
                break
            t = max(0, min(len(moves) - 1, t))
            m = moves[t]
            add(t, '中盘第 %d 手：%.0f%% 的模拟落在首选点，网络价值 v = %.3f'
                % (t + 1, m['confidence'] * 100, m['net_value']))

    chosen.sort(key=lambda c: c[0])
    return chosen[:5]


def build(src='demo_game.json', dst='demo_game.html', title='对局回放'):
    data_path = os.path.join(HERE, src)
    if not os.path.exists(data_path):
        raise SystemExit('先运行 demo_game.py 生成 %s' % src)
    with open(data_path, encoding='utf-8') as fh:
        G = json.load(fh)

    meta = G['meta']
    keys = pick_key_steps(G['moves'], meta['board_width'])
    G['key_steps'] = keys

    html = TEMPLATE.replace('__GAME__', json.dumps(G, separators=(',', ':')))
    html = html.replace('__TITLE__', title)
    out = os.path.join(HERE, dst)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('written:', out)
    print('plies:', meta['total_moves'], '| key steps:',
          [k[0] for k in keys])
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · AlphaZero-Gomoku</title>
<style>
:root{
  --bg:#141414; --card:#1c1c1c; --card2:#212120; --line:rgba(255,255,255,.09);
  --fg:#e8e8e6; --dim:#9a9a96; --dim2:#6f6f6b;
  --p1:#F0997B; --p2:#85B7EB; --prior:#6f6f6b; --accent:#EF9F27; --teal:#5DCAA5;
  --board:#1a1a19;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:"Noto Sans CJK SC","Source Han Sans CN","PingFang SC",system-ui,sans-serif;
  font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:36px 24px 64px}
h1{font-size:22px;font-weight:500;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12.5px;margin:0 0 22px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.chip{background:var(--card);border:.5px solid var(--line);border-radius:999px;
  padding:4px 12px;font-size:12px;color:var(--dim)}
.chip b{color:var(--fg);font-weight:500}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:22px;align-items:start}
@media(max-width:820px){.layout{grid-template-columns:1fr}}
.boardwrap{background:var(--card);border:.5px solid var(--line);border-radius:14px;padding:16px}
#board{display:grid;gap:0;width:100%;aspect-ratio:1;user-select:none}
.cell{position:relative;border-right:.5px solid rgba(255,255,255,.055);
  border-bottom:.5px solid rgba(255,255,255,.055)}
.cell.edge-r{border-right:none}.cell.edge-b{border-bottom:none}
.lab{display:flex;align-items:center;justify-content:center;
  font-size:10px;color:var(--dim2);font-variant-numeric:tabular-nums}
.stone{position:absolute;inset:8%;border-radius:50%;opacity:0;transform:scale(.4);
  transition:opacity .18s ease,transform .18s ease}
.stone.on{opacity:1;transform:scale(1)}
.stone.p1{background:var(--p1)}
.stone.p2{background:var(--p2)}
.stone.last{box-shadow:0 0 0 2px rgba(255,255,255,.85),0 0 0 4px rgba(0,0,0,.35)}
.cand{position:absolute;border-radius:50%;background:rgba(239,159,39,.20);
  border:1px solid rgba(239,159,39,.55);display:flex;align-items:center;
  justify-content:center;font-size:9.5px;color:#F5C97A;
  font-variant-numeric:tabular-nums;transition:opacity .15s}
.cand.hide{opacity:0}
.cand.best{border-width:1.5px;border-color:var(--accent);background:rgba(239,159,39,.34);color:#FFD79A}
.side{display:flex;flex-direction:column;gap:14px}
.panel{background:var(--card);border:.5px solid var(--line);border-radius:12px;padding:15px 16px}
.panel h3{font-size:12px;font-weight:400;color:var(--dim);margin:0 0 10px}
.now{display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
.now .ply{font-size:26px;font-weight:500;letter-spacing:-.02em}
.now .who{font-size:12.5px;color:var(--dim)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:12.5px}
.kv dt{color:var(--dim)}
.kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.cmp{display:grid;grid-template-columns:44px 1fr 1fr;gap:5px 8px;align-items:center;font-size:11.5px}
.cmp .hd{color:var(--dim2);font-size:10.5px;padding-bottom:2px}
.cmp .pos{color:var(--dim);font-variant-numeric:tabular-nums}
.bar{position:relative;height:13px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;border-radius:3px;display:block}
.bar span{position:absolute;inset:0;display:flex;align-items:center;
  padding-left:4px;font-size:9.5px;color:#dcdcd8;font-variant-numeric:tabular-nums}
.ctrl{display:flex;align-items:center;gap:8px;margin-top:16px}
button{background:var(--card2);color:var(--fg);border:.5px solid var(--line);
  border-radius:8px;padding:6px 12px;font-size:12.5px;cursor:pointer;font-family:inherit}
button:hover{border-color:rgba(255,255,255,.25)}
button:disabled{opacity:.35;cursor:default}
input[type=range]{flex:1;accent-color:var(--accent)}
.keys{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-top:26px}
.key{background:var(--card);border:.5px solid var(--line);border-radius:10px;
  padding:11px 13px;cursor:pointer;transition:border-color .15s}
.key:hover{border-color:rgba(239,159,39,.5)}
.key.active{border-color:var(--accent)}
.key .n{font-size:11px;color:var(--dim2);margin-bottom:3px}
.key .t{font-size:12.5px;line-height:1.45}
h2{font-size:13px;font-weight:500;color:var(--dim);margin:32px 0 10px;
  letter-spacing:.02em}
#vcurve{width:100%;height:86px;display:block}
.vlegend{display:flex;gap:14px;font-size:11px;color:var(--dim);margin-top:6px}
.vlegend i{display:inline-block;width:12px;height:2px;vertical-align:middle;margin-right:4px}
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub" id="sub"></p>
  <div class="chips" id="chips"></div>

  <div class="layout">
    <div>
      <div class="boardwrap"><div id="board"></div></div>
      <div class="ctrl">
        <button id="first">⏮</button>
        <button id="prev">◀</button>
        <button id="play">播放</button>
        <button id="next">▶</button>
        <button id="last">⏭</button>
        <input type="range" id="slider" min="0" max="0" value="0">
        <span id="pos" style="font-size:12px;color:var(--dim);min-width:56px;text-align:right"></span>
      </div>
    </div>

    <div class="side">
      <div class="panel">
        <div class="now">
          <span class="ply" id="ply">1</span>
          <span class="who" id="who"></span>
        </div>
        <dl class="kv">
          <dt>落点</dt><dd id="pos2">—</dd>
          <dt>网络价值 v</dt><dd id="nv">—</dd>
          <dt>搜索 Q</dt><dd id="rq">—</dd>
          <dt>首选点访问占比</dt><dd id="cf">—</dd>
          <dt>搜索量</dt><dd id="rv">—</dd>
        </dl>
      </div>

      <div class="panel">
        <h3>先验 vs 搜索后的概率</h3>
        <div class="cmp" id="cmp">
          <div class="hd"></div><div class="hd">网络先验</div><div class="hd">MCTS 搜索</div>
        </div>
      </div>

      <div class="panel">
        <h3>网络价值 v 与概率集中度</h3>
        <svg id="vcurve" viewBox="0 0 300 104" preserveAspectRatio="none"></svg>
        <div class="vlegend">
          <span><i style="background:var(--teal)"></i>价值 v（左轴 −1…+1）</span>
          <span><i style="background:var(--accent)"></i>首选访问占比</span>
        </div>
      </div>
    </div>
  </div>

  <h2>关键节点</h2>
  <div class="keys" id="keys"></div>
</div>

<script>
const G = __GAME__;
const M = G.meta, MV = G.moves;
const S = M.board_width, N = MV.length;

const boardEl = document.getElementById('board');
const cols = S + 1;
boardEl.style.gridTemplateColumns = `18px repeat(${S}, 1fr)`;
boardEl.style.gridTemplateRows = `18px repeat(${S}, 1fr)`;

const cellAt = {};
for (let hr = 0; hr <= S; hr++) {
  for (let cc = 0; cc <= S; cc++) {
    const d = document.createElement('div');
    const isLabRow = hr === 0, isLabCol = cc === 0;
    d.className = 'cell';
    if (isLabRow || isLabCol) d.classList.add('lab');
    else {
      if (cc === S) d.classList.add('edge-r');
      if (hr === S) d.classList.add('edge-b');
    }
    if (isLabRow && cc > 0) d.textContent = String(cc - 1);
    if (isLabCol && hr > 0) d.textContent = String(S - hr);
    boardEl.appendChild(d);
    if (!isLabRow && !isLabCol) cellAt[`${hr},${cc}`] = d;
  }
}

MV.forEach((m, i) => {
  const cell = cellAt[`${S - m.h},${m.w + 1}`];
  const st = document.createElement('div');
  st.className = 'stone ' + (m.player === 1 ? 'p1' : 'p2');
  st.id = 'st' + i;
  cell.appendChild(st);
});

const candEls = [];
function renderCands(i) {
  candEls.forEach(e => e.remove());
  candEls.length = 0;
  if (i < 0 || i >= N) return;
  const m = MV[i];
  const maxP = Math.max(...m.search.map(p => p[1]), 0.001);
  m.search.forEach(([mv, p], k) => {
    const h = Math.floor(mv / S), w = mv % S;
    const cell = cellAt[`${S - h},${w + 1}`];
    if (!cell) return;
    const s = document.createElement('div');
    const size = 26 + 46 * Math.sqrt(p / maxP);
    s.className = 'cand' + (k === 0 ? ' best' : '');
    s.style.width = size + '%';
    s.style.height = size + '%';
    s.style.left = (50 - size / 2) + '%';
    s.style.top = (50 - size / 2) + '%';
    s.textContent = p >= 0.02 ? Math.round(p * 100) + '' : '';
    cell.appendChild(s);
    candEls.push(s);
  });
}

function fmt(v, d) { return v === null || v === undefined ? '—' : v.toFixed(d === undefined ? 3 : d); }

function show(i) {
  for (let k = 0; k < N; k++) {
    const st = document.getElementById('st' + k);
    st.classList.toggle('on', k <= i);
    st.classList.toggle('last', k === i);
  }
  const m = i >= 0 ? MV[i] : null;
  document.getElementById('ply').textContent = i >= 0 ? (i + 1) : '—';
  const who = document.getElementById('who');
  if (m) {
    who.innerHTML = `<span class="dot" style="background:${m.player===1?'var(--p1)':'var(--p2)'}"></span>`
      + (m.player === 1 ? '先手' : '后手');
    document.getElementById('pos2').textContent = `(${m.w}, ${m.h})`;
    document.getElementById('nv').textContent = fmt(m.net_value);
    document.getElementById('rq').textContent = fmt(m.root_q);
    document.getElementById('cf').textContent = (m.confidence * 100).toFixed(1) + '%';
    document.getElementById('rv').textContent = m.root_visits;

    const cmp = document.getElementById('cmp');
    cmp.querySelectorAll('.row').forEach(e => e.remove());
    const priorMap = Object.fromEntries(m.prior);
    const rows = m.search;
    rows.forEach(([mv, p], k) => {
      const h = Math.floor(mv / S), w = mv % S;
      const pv = priorMap[mv] || 0;
      const mk = (v, color) => {
        const wrap = document.createElement('div');
        wrap.className = 'bar';
        const bar = document.createElement('i');
        bar.style.width = Math.max(1.2, v * 100) + '%';
        bar.style.background = color;
        const sp = document.createElement('span');
        sp.textContent = Math.round(v * 100) + '%';
        wrap.appendChild(bar); wrap.appendChild(sp);
        return wrap;
      };
      const a = document.createElement('div');
      a.className = 'pos row';
      a.style.gridColumn = '1';
      a.textContent = `(${w},${h})`;
      const b = mk(pv, 'var(--prior)');
      b.classList.add('row');
      const c = mk(p, 'var(--accent)');
      c.classList.add('row');
      cmp.appendChild(a); cmp.appendChild(b); cmp.appendChild(c);
    });
  }
  document.getElementById('slider').value = i;
  document.getElementById('pos').textContent = `${i + 1} / ${N}`;
  renderCands(i);
  document.querySelectorAll('.key').forEach(el => {
    el.classList.toggle('active', Number(el.dataset.i) === i);
  });
  document.getElementById('prev').disabled = i <= 0;
  document.getElementById('first').disabled = i <= 0;
  document.getElementById('next').disabled = i >= N - 1;
  document.getElementById('last').disabled = i >= N - 1;
}

let cur = 0;
document.getElementById('slider').max = N - 1;
document.getElementById('slider').addEventListener('input', e => { stop(); show(Number(e.target.value)); });
document.getElementById('prev').onclick = () => { stop(); show(Math.max(0, cur - 1)); };
document.getElementById('next').onclick = () => { stop(); show(Math.min(N - 1, cur + 1)); };
document.getElementById('first').onclick = () => { stop(); show(0); };
document.getElementById('last').onclick = () => { stop(); show(N - 1); };

let timer = null;
function stop() { if (timer) { clearInterval(timer); timer = null; } document.getElementById('play').textContent = '播放'; }
document.getElementById('play').onclick = () => {
  if (timer) { stop(); return; }
  document.getElementById('play').textContent = '暂停';
  if (cur >= N - 1) show(0);
  timer = setInterval(() => {
    if (cur >= N - 1) { stop(); return; }
    show(cur + 1);
  }, 900);
};

const keysEl = document.getElementById('keys');
G.key_steps.forEach(([i, reason]) => {
  const m = MV[i];
  const d = document.createElement('div');
  d.className = 'key'; d.dataset.i = i;
  d.innerHTML = `<div class="n">第 ${i + 1} 手 · ${m.player === 1 ? '先手' : '后手'} (${m.w}, ${m.h})</div>
    <div class="t">${reason}</div>`;
  d.onclick = () => { stop(); show(i); };
  keysEl.appendChild(d);
});

const svg = document.getElementById('vcurve');
const W = 300, H = 104, PAD = 6;
const maxConf = Math.max(...MV.map(m => Math.max(m.confidence,
  (m.prior.length ? m.prior[0][1] : 0))), 0.01);
const xAt = i => PAD + (W - 2 * PAD) * (N <= 1 ? 0 : i / (N - 1));
const poly = (yFn, color, w, dash) => {
  const pts = MV.map((m, i) => `${xAt(i).toFixed(1)},${yFn(m, i).toFixed(1)}`).join(' ');
  const pl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  pl.setAttribute('points', pts);
  pl.setAttribute('fill', 'none');
  pl.setAttribute('stroke', color);
  pl.setAttribute('stroke-width', w);
  pl.setAttribute('stroke-linejoin', 'round');
  if (dash) pl.setAttribute('stroke-dasharray', dash);
  svg.appendChild(pl);
};
const zero = document.createElementNS('http://www.w3.org/2000/svg', 'line');
zero.setAttribute('x1', PAD); zero.setAttribute('x2', W - PAD);
zero.setAttribute('y1', H / 2); zero.setAttribute('y2', H / 2);
zero.setAttribute('stroke', 'rgba(255,255,255,.18)');
zero.setAttribute('stroke-dasharray', '3,3');
svg.appendChild(zero);

poly(m => H / 2 - (H / 2 - PAD) * Math.max(-1, Math.min(1, m.net_value)), '#5DCAA5', 1.8);
poly(m => H - PAD - (H - 2 * PAD) * Math.min(1, m.confidence / maxConf), '#EF9F27', 1.2, '4,3');

document.getElementById('sub').textContent =
  `${M.model} · ${M.backend} · 每步 ${M.n_playout} 次模拟 · ${M.total_moves} 手 · ${M.elapsed}s`;
document.getElementById('chips').innerHTML = [
  ['棋盘', `${M.board_width}×${M.board_height}`],
  ['连子', `${M.n_in_row} 连`],
  ['结果', M.winner === -1 ? '平局' : (M.winner === 1 ? '先手胜' : '后手胜')],
  ['c_puct', M.c_puct],
  ['随机种子', M.seed],
].map(([k, v]) => `<div class="chip">${k} <b>${v}</b></div>`).join('');

show(0);
</script>
</body>
</html>
"""


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='demo_game.json')
    ap.add_argument('--output', default='demo_game.html')
    ap.add_argument('--title', default='对局回放')
    a = ap.parse_args()
    build(a.input, a.output, a.title)
