# -*- coding: utf-8 -*-
"""图形化人机对战：本地 HTTP 服务 + 浏览器棋盘。

    python web_play.py                 # 打开 http://127.0.0.1:8777
    python web_play.py --playout 800 --model best_policy.model
    python web_play.py --port 9000 --no-browser

设计：MCTS 留在 Python 进程里（模型是 torch/numpy 的，浏览器跑不动），
前端只做渲染与交互。所有对局状态由服务端单例 Session 持有，改动走
HTTP API，因此刷新页面不会丢局。

为什么不用 tkinter/pygame：这台机器没有可靠的显示环境，Web 方案还能
顺带在手机/另一台机器上看。
"""

from __future__ import print_function

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import Board                                    # noqa: E402
from mcts_alphaZero import MCTS                            # noqa: E402
from demo_game import load_backend, make_policy_fn          # noqa: E402


# 顺序即前端下拉框顺序，实测棋力降序
MODEL_PREFERENCE = (
    'best_policy_8_8_5.model',
    'best_policy.model',
    'best_policy_6_6_4.model',
    'best_policy_8_8_5.model2',
    'best_policy.model2',
    'best_policy_6_6_4.model2',
    'ablate10_policy.model',
    'fixed8_policy.model',
    'fixed_policy.model',
)


def list_models():
    seen, out = set(), []
    for n in MODEL_PREFERENCE:
        if os.path.exists(os.path.join(HERE, n)) and n not in seen:
            out.append(n)
            seen.add(n)
    for n in sorted(os.listdir(HERE)):
        if n.endswith('.model') and n not in seen:
            out.append(n)
            seen.add(n)
    return out


# ---------------------------------------------------------------- 棋局逻辑

def find_win_line(board, winner):
    """找出获胜的那 n 个连子，用于前端高亮。"""
    w, n, states = board.width, board.n_in_row, board.states
    for m, p in states.items():
        if p != winner:
            continue
        h, c = divmod(m, w)
        for dh, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            line = [m]
            for k in range(1, n):
                hh, cc = h + dh * k, c + dc * k
                if 0 <= hh < board.height and 0 <= cc < w \
                        and states.get(hh * w + cc) == winner:
                    line.append(hh * w + cc)
                else:
                    break
            if len(line) == n:
                return line
    return []


# ---------------------------------------------------------------- AI

class Brain(object):
    """把网络包成「给棋盘返回一步棋 + 决策依据」。"""

    def __init__(self, model_name, playout, c_puct):
        path = os.path.join(HERE, model_name)
        net, side, n_in_row, backend = load_backend(path)
        self.model = model_name
        self.side = side
        self.n_in_row = n_in_row
        self.backend = backend
        self.playout = playout
        self.c_puct = c_puct
        self.policy_fn = make_policy_fn(net)

    def choose(self, board):
        t0 = time.time()
        prior_pairs, net_value = self.policy_fn(board)
        prior_pairs = list(prior_pairs)

        mcts = MCTS(self.policy_fn, c_puct=self.c_puct,
                    n_playout=self.playout)
        acts, probs = mcts.get_move_probs(board, temp=1e-3)

        visits = {a: int(mcts._root._children[a]._n_visits) for a in acts}
        qvals = {a: float(mcts._root._children[a]._Q) for a in acts}
        visited = sum(visits.values()) or 1
        move = max(acts, key=lambda a: visits[a])
        h, w = divmod(move, board.width)

        # 注意：不能拿 get_move_probs 返回的 probs 当展示概率。temp=1e-3 时
        # softmax(1000*log(visits)) 已经是一张 one-hot，除首选外全是 0。
        # 展示用访问数占比（即 AlphaGo Zero 里 temp=1 的 π），这才是真实分布。
        def pack(a):
            return {'move': int(a), 'h': int(a // board.width),
                    'w': int(a % board.width),
                    'p': round(visits[a] / visited, 4),
                    'n': visits[a], 'q': round(qvals[a], 4)}

        cands = [pack(a) for a in acts if visits[a] > 0]
        cands.sort(key=lambda d: -d['n'])
        n_cands = len(cands)

        return {
            'move': move, 'h': h, 'w': w,
            'value': round(float(net_value), 4),
            'elapsed': round(time.time() - t0, 2),
            'visits': int(sum(visits.values())),
            'n_cands': n_cands,
            'top': cands[:3],
            'prior': sorted(
                ({'move': int(a), 'h': int(a // board.width),
                  'w': int(a % board.width), 'p': round(float(p), 4)}
                 for a, p in prior_pairs),
                key=lambda d: -d['p'])[:3],
            'cands': cands[:48],
        }


# ---------------------------------------------------------------- Session

class Session(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.brains = {}
        self.model = None
        self.playout = 400
        self.c_puct = 5.0
        self.human_first = True
        self.side = 8
        self.n_in_row = 5
        self.board = None
        self.history = []
        self.ai_last = None
        self.hint = None
        self.status = 'playing'
        self.win_line = []
        self.message = ''

    # -- 模型缓存：切换模型不该重训，但加载要几秒，缓存住
    def brain(self, name, playout=None):
        key = (name, playout or self.playout)
        if key not in self.brains:
            self.brains[key] = Brain(name, key[1], self.c_puct)
        return self.brains[key]

    def reset(self, model=None, playout=None, human_first=None):
        if model:
            self.model = model
        if playout:
            self.playout = int(playout)
        if human_first is not None:
            self.human_first = bool(human_first)
        if not self.model:
            self.model = list_models()[0]

        b = self.brain(self.model)
        self.side, self.n_in_row = b.side, b.n_in_row
        self.board = Board(width=self.side, height=self.side,
                           n_in_row=self.n_in_row)
        self.board.init_board(0)          # player1 恒先手，谁执子由 human_first 定
        if self.human_first:
            self.human_player, self.ai_player = 1, 2
        else:
            self.human_player, self.ai_player = 2, 1

        self.history, self.ai_last, self.hint = [], None, None
        self.status, self.win_line, self.message = 'playing', [], ''
        return self.snapshot()

    def _rebuild(self):
        self.board = Board(width=self.side, height=self.side,
                           n_in_row=self.n_in_row)
        self.board.init_board(0)
        for mv in self.history:
            self.board.do_move(mv)

    def _settle(self):
        end, winner = self.board.game_end()
        if not end:
            self.status = 'playing'
            self.win_line = []
            return False
        self.win_line = find_win_line(self.board, winner) if winner != -1 else []
        if winner == -1:
            self.status = 'tie'
        elif winner == self.human_player:
            self.status = 'human_win'
        else:
            self.status = 'ai_win'
        return True

    def human_move(self, move):
        if self.status != 'playing':
            raise ValueError('这一局已经结束了，点「重新开始」')
        if self.board.current_player != self.human_player:
            raise ValueError('还没轮到你')
        if move not in self.board.availables:
            raise ValueError('这个点不能落子')
        self.board.do_move(move)
        self.history.append(move)
        self.hint = None
        self._settle()
        return self.snapshot()

    def ai_move(self):
        if self.status != 'playing':
            raise ValueError('这一局已经结束了')
        if self.board.current_player != self.ai_player:
            raise ValueError('还没轮到 AI')
        res = self.brain(self.model).choose(self.board)
        self.board.do_move(res['move'])
        self.history.append(res['move'])
        self.ai_last = res
        self._settle()
        return self.snapshot()

    def make_hint(self):
        if self.status != 'playing' or \
                self.board.current_player != self.human_player:
            raise ValueError('现在不能提示')
        b = self.brain(self.model)
        old = b.playout
        b.playout = max(200, self.playout // 2)
        try:
            res = b.choose(self.board)
        finally:
            b.playout = old
        self.hint = res
        return self.snapshot()

    def undo(self):
        if not self.history:
            raise ValueError('没有可撤销的落子')
        for _ in range(4):
            if not self.history:
                break
            self.history.pop()
            self._rebuild()
            if not self.history or \
                    self.board.current_player == self.human_player:
                break
        self.ai_last, self.hint = None, None
        self.status, self.win_line = 'playing', []
        return self.snapshot()

    # -- 序列化
    def snapshot(self):
        side = self.side
        rows = []
        for h in range(side - 1, -1, -1):
            rows.append([int(self.board.states.get(h * side + w, 0))
                         for w in range(side)])
        need_ai = (self.status == 'playing'
                   and self.board.current_player == self.ai_player)
        return {
            'model': self.model,
            'backend': self.brain(self.model).backend,
            'playout': self.playout,
            'c_puct': self.c_puct,
            'side': side,
            'n_in_row': self.n_in_row,
            'board': rows,
            'last_move': int(self.board.last_move),
            'current_player': int(self.board.current_player),
            'human_player': self.human_player,
            'ai_player': self.ai_player,
            'human_first': self.human_first,
            'status': self.status,
            'need_ai': need_ai,
            'win_line': self.win_line,
            'history': self.history,
            'ai_last': self.ai_last,
            'hint': self.hint,
            'moves': len(self.history),
        }


SESSION = Session()


# ---------------------------------------------------------------- HTTP

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaZero 五子棋 · 人机对战</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#14161a; --panel:#1b1e24; --panel2:#20242c; --line:#2b313b;
  --text:#e7eaf0; --dim:#8b93a1; --dim2:#5d6573;
  --ai:#4ea1ff; --human:#ffab4a; --ok:#45d6a4; --bad:#ff6b6b;
}
html,body{height:100%}
body{background:var(--bg);color:var(--text);
  font:14px/1.5 -apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;
  -webkit-font-smoothing:antialiased;overflow:hidden}
.app{display:flex;height:100vh;gap:18px;padding:18px}
.left{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:14px;min-width:0}
.boardwrap{position:relative;display:flex;align-items:center;justify-content:center}
canvas{border-radius:14px;box-shadow:0 18px 50px rgba(0,0,0,.55);
  cursor:pointer;display:block}
canvas.waiting{cursor:progress}
.overlay{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;border-radius:14px;background:rgba(10,12,15,.55);
  backdrop-filter:blur(2px);pointer-events:none;opacity:0;
  transition:opacity .18s}
.overlay.on{opacity:1}
.spin{width:26px;height:26px;border:3px solid rgba(255,255,255,.22);
  border-top-color:#fff;border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.statusbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  justify-content:center;color:var(--dim);font-size:13px;min-height:22px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;
  border-radius:999px;background:var(--panel);border:1px solid var(--line)}
.dot{width:11px;height:11px;border-radius:50%;flex:none;
  box-shadow:0 0 0 1px rgba(255,255,255,.14) inset}
.dot.black{background:radial-gradient(circle at 33% 30%,#5b6472,#0b0d11 72%)}
.dot.white{background:radial-gradient(circle at 33% 30%,#fff,#b9bec8 78%)}
.side{width:330px;flex:none;display:flex;flex-direction:column;gap:12px;
  overflow-y:auto;padding-right:2px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:13px 14px}
.card h3{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--dim2);font-weight:600;margin-bottom:11px}
.row{display:flex;align-items:center;justify-content:space-between;
  gap:10px;margin-bottom:9px;font-size:13px}
.row:last-child{margin-bottom:0}
.row label{color:var(--dim)}
select,input[type=range]{background:var(--panel2);color:var(--text);
  border:1px solid var(--line);border-radius:7px;padding:5px 7px;font:inherit;
  font-size:12.5px;outline:none;max-width:190px}
select:focus{border-color:var(--ai)}
input[type=range]{padding:0;height:4px;-webkit-appearance:none;background:#2b313b;
  border:none;width:130px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;
  height:13px;border-radius:50%;background:var(--ai);cursor:pointer}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);
  border-radius:8px;padding:8px 10px;font:inherit;font-size:13px;cursor:pointer;
  transition:.15s}
button:hover:not(:disabled){background:#2a3038;border-color:#3b434f}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:linear-gradient(180deg,#2563c9,#1d4fa5);
  border-color:#2b6ad0;color:#fff}
button.primary:hover:not(:disabled){filter:brightness(1.12)}
.meter{height:9px;border-radius:99px;background:#22262e;overflow:hidden;
  position:relative;margin:2px 0 7px}
.meter .fill{height:100%;width:50%;border-radius:99px;
  background:linear-gradient(90deg,#2b6ad0,#4ea1ff);transition:width .35s
  cubic-bezier(.4,0,.2,1),background .35s}
.meter .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:rgba(255,255,255,.18)}
.meter .fill.bad{background:linear-gradient(90deg,#c94a2b,#ff8461)}
.meter .fill.good{background:linear-gradient(90deg,#1f9e6f,#45d6a4)}
.meta{display:flex;justify-content:space-between;font-size:12px;
  color:var(--dim2)}
.cand{display:flex;align-items:center;gap:8px;font-size:12.5px;
  padding:4px 0;color:var(--dim)}
.cand b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums;
  min-width:44px}
.bar{flex:1;height:5px;background:#22262e;border-radius:99px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ai);border-radius:99px}
.movelist{max-height:33vh;overflow-y:auto;margin:-3px -4px 0;
  padding:3px 4px}
.mv{display:grid;grid-template-columns:30px 26px 1fr auto;gap:8px;
  align-items:center;font-size:12.5px;padding:3px 6px;border-radius:6px;
  color:var(--dim);font-variant-numeric:tabular-nums}
.mv:nth-child(odd){background:rgba(255,255,255,.022)}
.mv.last{background:rgba(78,161,255,.13);color:var(--text)}
.mv .n{color:var(--dim2)}
.mv .st{width:11px;height:11px;border-radius:50%}
.mv .t{color:var(--dim2);font-size:11.5px}
.empty{color:var(--dim2);font-size:12.5px;text-align:center;padding:12px 0}
#toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,14px);
  background:#2a2f38;border:1px solid #3b434f;color:#fff;padding:9px 16px;
  border-radius:9px;font-size:13px;opacity:0;transition:.25s;
  pointer-events:none;box-shadow:0 10px 30px rgba(0,0,0,.5)}
#toast.on{opacity:1;transform:translate(-50%,0)}
.tag{font-size:11px;padding:1px 6px;border-radius:5px;background:#22262e;
  color:var(--dim);margin-left:6px}
@media (max-width:980px){
  body{overflow:auto}
  .app{flex-direction:column;height:auto}
  .side{width:100%;max-width:560px;margin:0 auto}
}
</style></head><body>
<div class="app">
  <div class="left">
    <div class="boardwrap">
      <canvas id="cv" width="560" height="560"></canvas>
      <div class="overlay" id="ov"><div class="spin"></div></div>
    </div>
    <div class="statusbar" id="sb"></div>
  </div>

  <aside class="side">
    <section class="card">
      <h3>对局设置</h3>
      <div class="row"><label>模型</label><select id="selModel"></select></div>
      <div class="row"><label>模拟次数</label>
        <span><input type="range" id="rngPlayout" min="100" max="1600"
          step="100" value="400"><span id="lblPlayout"
          style="color:var(--text);margin-left:8px;font-size:12.5px;
          font-variant-numeric:tabular-nums">400</span></span></div>
      <div class="row"><label>先手</label>
        <select id="selFirst">
          <option value="human">我执黑先手</option>
          <option value="ai">AI 执黑先手</option>
        </select></div>
      <div class="btns">
        <button class="primary" id="btnNew">重新开始</button>
        <button id="btnUndo">撤销一手</button>
        <button id="btnHint">AI 提示</button>
        <button id="btnHeat">显示：搜索分布</button>
      </div>
    </section>

    <section class="card">
      <h3>AI 形势判断</h3>
      <div class="meter"><div class="fill" id="mFill"></div>
        <div class="mid"></div></div>
      <div class="meta"><span id="mLeft">—</span><span id="mRight">—</span></div>
      <div style="margin-top:12px" id="cands"></div>
    </section>

    <section class="card">
      <h3>落子记录 <span class="tag" id="mvCount">0</span></h3>
      <div class="movelist" id="mvList"><div class="empty">还没有落子</div></div>
    </section>
  </aside>
</div>
<div id="toast"></div>

<script>/*__BOOT__*/</script>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const ov = document.getElementById('ov');
let S = null, busy = false, view = 'heat', hover = null, px = 0;

const COL = 'ABCDEFGHJKLMNOPQRST';

function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('on');
  clearTimeout(t._h); t._h = setTimeout(()=>t.classList.remove('on'), 2200);
}

async function api(path, body){
  // credentials 显式写上：局域网访问靠 cookie 认人，默认值在旧浏览器里是 omit
  const opt = body === undefined
    ? {credentials: 'same-origin'}
    : {method:'POST', headers:{'Content-Type':'application/json'},
       body: JSON.stringify(body), credentials: 'same-origin'};
  try{
    const r = await fetch(path, opt);
    const j = await r.json();
    if(j.error) toast(j.error);
    return j;
  }catch(e){ toast('后端没有响应'); return null; }
}

/* ---------------- 绘制 ---------------- */
function fit(side){
  // 量 .left 而不是 .boardwrap：canvas 在 wrap 里，用 wrap.clientWidth
  // 会读到 canvas 自己的宽度，形成「越画越小」的反馈回路。
  const host = document.querySelector('.left');
  const avail = Math.min((host.clientWidth || 560) - 4,
                         window.innerHeight - 110, 660);
  const size = Math.max(260, Math.floor(avail));
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = size + 'px'; cv.style.height = size + 'px';
  cv.width = Math.round(size * dpr); cv.height = Math.round(size * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  px = size;
  return size;
}

function geom(side, size){
  const cell = size / (side + 1);
  return {cell, pad: cell, size};
}

function pt(g, side, h, w){
  return [g.pad + w * g.cell, g.pad + (side - 1 - h) * g.cell];
}

function draw(){
  if(!S) return;
  const side = S.side, size = fit(side), g = geom(side, size);
  const R = g.cell * 0.44;

  // 木质底色
  const bg = ctx.createLinearGradient(0, 0, size, size);
  bg.addColorStop(0, '#e8c08a'); bg.addColorStop(.45, '#dcae74');
  bg.addColorStop(1, '#c99a60');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, size, size);

  // 木纹
  ctx.save(); ctx.globalAlpha = .05; ctx.strokeStyle = '#5b3a17';
  ctx.lineWidth = 1;
  for(let i = 0; i < 26; i++){
    ctx.beginPath();
    const y = (i + .5) * size / 26;
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(size * .3, y + 5, size * .65, y - 6, size, y + 2);
    ctx.stroke();
  }
  ctx.restore();

  // 网格
  ctx.strokeStyle = 'rgba(72,48,22,.55)'; ctx.lineWidth = 1;
  for(let i = 0; i < side; i++){
    const p = g.pad + i * g.cell;
    ctx.beginPath(); ctx.moveTo(g.pad, p); ctx.lineTo(size - g.pad, p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, g.pad); ctx.lineTo(p, size - g.pad); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(72,48,22,.9)'; ctx.lineWidth = 1.6;
  ctx.strokeRect(g.pad, g.pad, size - 2 * g.pad, size - 2 * g.pad);

  // 座标
  ctx.fillStyle = 'rgba(72,48,22,.62)';
  ctx.font = '600 11px ui-monospace,monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for(let i = 0; i < side; i++){
    const p = g.pad + i * g.cell;
    ctx.fillText(COL[i], p, g.pad * .42);
    ctx.fillText(String(i + 1), g.pad * .4, size - g.pad - i * g.cell);
  }

  // 搜索分布：只画前 8 个点。10x10 那份模型会把访问数摊到 40+ 个点上，
  // 全画出来就是一片圆点噪声，看不出主次。
  if(view === 'heat' && S.status === 'playing' && S.ai_last){
    const cs = S.ai_last.cands.slice(0, 8);
    const mx = Math.max(...cs.map(c => c.n));
    cs.forEach((c, i) => {
      if(S.board[side - 1 - c.h][c.w]) return;
      const [x, y] = pt(g, side, c.h, c.w);
      const share = c.n / mx;
      ctx.beginPath(); ctx.arc(x, y, R * (.17 + .3 * share), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(24,86,196,${(0.18 + 0.5 * share).toFixed(3)})`;
      ctx.fill();
      if(i === 0){
        ctx.beginPath(); ctx.arc(x, y, R * .66, 0, Math.PI * 2);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = 'rgba(24,86,196,.5)'; ctx.lineWidth = 1.3;
        ctx.stroke(); ctx.setLineDash([]);
      }
    });
  }

  // 提示点
  if(view === 'hint' && S.status === 'playing' && S.hint && S.hint.top.length){
    const mx = Math.max(...S.hint.top.map(c => c.p));
    S.hint.top.forEach((c, i) => {
      const [x, y] = pt(g, side, c.h, c.w);
      const r = R * (0.55 + 0.45 * c.p / mx);
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,150,40,.85)'; ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 12px -apple-system,sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(i + 1), x, y + .5);
    });
  }

  // 棋子
  for(let h = 0; h < side; h++){
    for(let w = 0; w < side; w++){
      const p = S.board[side - 1 - h][w];
      if(!p) continue;
      const [x, y] = pt(g, side, h, w);
      const black = (p === 1);
      ctx.beginPath(); ctx.arc(x + 1, y + 2.5, R, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(60,35,10,.28)'; ctx.fill();

      const gd = ctx.createRadialGradient(
        x - R * .35, y - R * .4, R * .12, x, y, R * 1.06);
      if(black){
        gd.addColorStop(0, '#5c6674'); gd.addColorStop(.42, '#23282f');
        gd.addColorStop(1, '#080a0d');
      }else{
        gd.addColorStop(0, '#ffffff'); gd.addColorStop(.55, '#f2efe8');
        gd.addColorStop(1, '#b6bcc6');
      }
      ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2);
      ctx.fillStyle = gd; ctx.fill();
      ctx.lineWidth = .8;
      ctx.strokeStyle = black ? 'rgba(0,0,0,.55)' : 'rgba(120,124,132,.5)';
      ctx.stroke();
    }
  }

  // 最后一手
  if(S.last_move >= 0 && S.status === 'playing'){
    const h = Math.floor(S.last_move / side), w = S.last_move % side;
    const [x, y] = pt(g, side, h, w);
    ctx.beginPath(); ctx.arc(x, y, R * .28, 0, Math.PI * 2);
    ctx.fillStyle = (S.board[side - 1 - h][w] === 1) ? '#ff8f3f' : '#d2452b';
    ctx.fill();
  }

  // 五连高亮
  if(S.win_line && S.win_line.length){
    ctx.save();
    ctx.strokeStyle = 'rgba(255,64,64,.95)'; ctx.lineWidth = Math.max(3, g.cell * .12);
    ctx.lineCap = 'round'; ctx.shadowColor = 'rgba(255,60,60,.8)';
    ctx.shadowBlur = 14;
    const a = pt(g, side, ...divmod(S.win_line[0], side));
    const b = pt(g, side, ...divmod(S.win_line[S.win_line.length - 1], side));
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.restore();
  }

  // 悬停虚影
  if(hover && S.status === 'playing' && !busy
     && S.current_player === S.human_player){
    const mv = hover.h * side + hover.w;
    if(mv >= 0 && !S.board[side - 1 - hover.h][hover.w]){
      const [x, y] = pt(g, side, hover.h, hover.w);
      ctx.globalAlpha = .3;
      ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2);
      ctx.fillStyle = (S.human_player === 1) ? '#12151a' : '#f4f1ea';
      ctx.fill(); ctx.globalAlpha = 1;
    }
  }
}

function divmod(a, b){ return [Math.floor(a / b), a % b]; }

/* ---------------- 侧栏 ---------------- */
function renderSide(){
  const sb = document.getElementById('sb');
  const mine = S.current_player === S.human_player;
  const over = S.status !== 'playing';
  let whom, mark;
  if(over){
    whom = {human_win:'你赢了', ai_win:'AI 赢了', tie:'平局'}[S.status];
    mark = '';
  }else{
    whom = mine ? '轮到你' : 'AI 思考中';
    mark = `<span class="dot ${S.current_player === 1 ? 'black' : 'white'}"></span>`;
  }
  sb.innerHTML =
    `<span class="chip">${mark}${whom}</span>` +
    `<span class="chip">${over ? '共 ' + S.moves + ' 手'
                              : '第 ' + (S.moves + 1) + ' 手'}</span>` +
    `<span class="chip">${S.side}×${S.side} · ${S.n_in_row} 连</span>` +
    `<span class="chip" title="后端">${S.backend}</span>`;

  const al = S.ai_last;
  const empties = S.board.reduce((a, r) => a + r.filter(v => !v).length, 0);
  const mf = document.getElementById('mFill');
  if(al){
    const wr = Math.max(0, Math.min(1, (al.value + 1) / 2));
    mf.style.width = (wr * 100).toFixed(1) + '%';
    mf.className = 'fill' + (wr > .62 ? ' good' : (wr < .38 ? ' bad' : ''));
    document.getElementById('mLeft').textContent =
      `AI 自评 ${al.value >= 0 ? '+' : ''}${al.value.toFixed(3)}`;
    document.getElementById('mRight').textContent =
      `${al.visits} 次模拟 · ${al.elapsed.toFixed(1)}s`;
    document.getElementById('cands').innerHTML = al.top.map(c =>
      `<div class="cand"><b>${COL[c.w]}${c.h + 1}</b>
        <div class="bar"><i style="width:${Math.max(2, c.p * 100).toFixed(1)}%"></i></div>
        <span style="min-width:40px;text-align:right">${(c.p * 100).toFixed(1)}%</span>
        <span style="min-width:34px;text-align:right;color:var(--dim2)">
        ${c.n}访</span></div>`).join('') +
      `<div class="meta" style="margin-top:9px">
        <span>进入搜索 ${al.n_cands} / ${empties} 点</span>
        <span>前三占 ${(al.top.reduce((a, c) => a + c.p, 0) * 100).toFixed(0)}%</span>
      </div>`;
  }else{
    mf.style.width = '50%'; mf.className = 'fill';
    document.getElementById('mLeft').textContent = '还没有 AI 评估';
    document.getElementById('mRight').textContent = '';
    document.getElementById('cands').innerHTML = '';
  }

  document.getElementById('mvCount').textContent = S.moves;
  const ml = document.getElementById('mvList');
  if(!S.history.length){
    ml.innerHTML = '<div class="empty">还没有落子</div>';
  }else{
    const extra = {};
    if(al) extra[S.history.length - 1] = al.elapsed;
    ml.innerHTML = S.history.map((mv, i) => {
      const h = Math.floor(mv / S.side), w = mv % S.side;
      // player1 先手，所以按奇偶还原执子方：偶数手是 player1
      const who = ((i % 2 === 0) === (S.human_player === 1))
        ? S.human_player : S.ai_player;
      const mark = who === 1 ? '#1a1e25' : '#eeeae2';
      return `<div class="mv ${i === S.history.length - 1 ? 'last' : ''}">
        <span class="n">${i + 1}</span>
        <span class="st" style="background:${mark};
          box-shadow:0 0 0 1px rgba(255,255,255,.18) inset"></span>
        <span>${COL[w]}${h + 1}${who === S.human_player
          ? '' : ' <span class="tag">AI</span>'}</span>
        <span class="t">${extra[i] ? extra[i].toFixed(1) + 's' : ''}</span>
      </div>`;
    }).reverse().join('');
  }

  document.getElementById('btnUndo').disabled = S.moves === 0 || busy;
  document.getElementById('btnHint').disabled =
    S.status !== 'playing' || busy || S.current_player !== S.human_player;
  document.getElementById('btnNew').disabled = busy;
  document.getElementById('btnHeat').textContent = VIEW_LABEL[view];
}

const VIEW_LABEL = {heat:'叠加：搜索分布', hint:'叠加：AI 提示', none:'叠加：关闭'};
const VIEW_NEXT = {heat:'none', none:'hint', hint:'heat'};

function render(){
  draw();
  renderSide();
}

/* ---------------- 交互 ---------------- */
function cellAt(ev){
  if(!S) return null;
  const side = S.side, g = geom(side, px);
  const r = cv.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const w = Math.round((x - g.pad) / g.cell);
  const row = Math.round((y - g.pad) / g.cell);
  const h = side - 1 - row;
  if(w < 0 || w >= side || h < 0 || h >= side) return null;
  return {h, w};
}

cv.addEventListener('mousemove', ev => {
  hover = cellAt(ev); draw();
});
cv.addEventListener('mouseleave', () => { hover = null; draw(); });
cv.addEventListener('click', async ev => {
  if(!S || busy || S.status !== 'playing') return;
  if(S.current_player !== S.human_player) return;
  const c = cellAt(ev);
  if(!c) return;
  const move = c.h * S.side + c.w;
  if(S.board[S.side - 1 - c.h][c.w]) return;
  await send('/api/move', {move: move});
});

async function send(path, body){
  busy = true; ov.classList.add('on'); cv.classList.add('waiting');
  renderSide();
  const j = await api(path, body);
  if(j && !j.error) S = j;
  busy = false; ov.classList.remove('on'); cv.classList.remove('waiting');
  render();
  if(S && S.need_ai) await aiTurn();
}

async function aiTurn(){
  busy = true; ov.classList.add('on'); cv.classList.add('waiting');
  renderSide();
  const j = await api('/api/ai', {});
  if(j && !j.error) S = j;
  busy = false; ov.classList.remove('on'); cv.classList.remove('waiting');
  render();
}

async function newGame(){
  const model = document.getElementById('selModel').value;
  const playout = +document.getElementById('rngPlayout').value;
  const first = document.getElementById('selFirst').value;
  S = await api('/api/new', {model, playout, human_first: first === 'human'});
  render();
  if(S && S.need_ai) await aiTurn();
}

document.getElementById('btnNew').onclick = newGame;
document.getElementById('btnUndo').onclick = () => send('/api/undo', {});
document.getElementById('btnHint').onclick = async () => {
  view = 'hint';
  await send('/api/hint', {});
};
document.getElementById('btnHeat').onclick = () => {
  let next = VIEW_NEXT[view];
  if(next === 'hint' && !S.hint) next = 'heat';
  view = next;
  render();
};
document.getElementById('rngPlayout').oninput = e => {
  document.getElementById('lblPlayout').textContent = e.target.value;
};

window.addEventListener('resize', () => { draw(); });

/* ---------------- 启动 ---------------- */
(async function init(){
  // 服务端已经把首屏状态内联进来了，直接画，避免空棋盘闪一下
  const sel = document.getElementById('selModel');
  const models = window.__MODELS__ || ((await api('/api/models')) || {}).models;
  if(models){
    sel.innerHTML = models.map(n => `<option>${n}</option>`).join('');
  }
  S = window.__BOOT__ || await api('/api/state');
  if(!S) return;
  if(S.model) sel.value = S.model;
  document.getElementById('rngPlayout').value = S.playout;
  document.getElementById('lblPlayout').textContent = S.playout;
  document.getElementById('selFirst').value = S.human_first ? 'human' : 'ai';
  render();
  if(S.need_ai) await aiTurn();
})();
</script>
</body></html>
"""


def render_page():
    """把当前局面内联进 HTML，首屏不需要等两个请求回来才有棋看。"""
    with SESSION.lock:
        boot = json.dumps(SESSION.snapshot(), ensure_ascii=False)
    models = json.dumps(list_models(), ensure_ascii=False)
    script = ('window.__BOOT__=%s;window.__MODELS__=%s;' % (boot, models))
    return HTML.replace('/*__BOOT__*/', script).encode('utf-8')


def lan_addresses():
    """返回 [(网卡名, IPv4)]，WiFi 优先。

    不能用「UDP connect 看出口地址」来决定：装了 Clash/TUN 类代理时，出口
    会是 198.18.x.x 这种假地址，手机根本连不上。改读网卡表，按名字排序。
    """
    skip = ('docker', 'br-', 'virbr', 'veth', 'tun', 'tap', 'wg', 'zt',
            'FlClash', 'utun', 'lo')
    found = []
    try:
        raw = subprocess.run(
            ['ip', '-4', '-o', 'addr', 'show', 'scope', 'global'],
            capture_output=True, text=True, timeout=3).stdout
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            ifname, ip = parts[1], parts[3].split('/')[0]
            if ifname.startswith(skip) or ip.startswith('198.18.'):
                continue
            found.append((ifname, ip))
    except (OSError, subprocess.SubprocessError):
        pass

    if not found:                       # 没有 iproute2 时的降级
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('223.5.5.5', 53))
            found.append(('?', s.getsockname()[0]))
        except OSError:
            pass
        finally:
            s.close()

    def rank(item):
        name = item[0]
        if name.startswith('wl'):
            return 0
        if name.startswith(('en', 'eth')):
            return 1
        return 2

    return sorted(found, key=rank)


TOKEN = None          # 非回环访问需要的令牌，main() 里赋值


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def _from_loopback(self):
        return self.client_address[0] in ('127.0.0.1', '::1')

    def _token_ok(self):
        """本机访问一律放行；局域网访问需要 ?t=令牌 或已种下的 cookie。"""
        if not TOKEN or self._from_loopback():
            return True
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if q.get('t') == [TOKEN]:
            return True
        for part in (self.headers.get('Cookie') or '').split(';'):
            k, _, v = part.strip().partition('=')
            if k == 'gp_tok' and v == TOKEN:
                return True
        return False

    def _send(self, code, ctype, payload, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, code=200):
        self._send(code, 'application/json; charset=utf-8',
                   json.dumps(obj, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        if not self._token_ok():
            self._json({'error': '需要访问令牌，请用启动时打印的完整链接打开'}, 401)
            return
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            extra = {}
            if TOKEN and not self._from_loopback():
                # 首屏带令牌进来之后种 cookie，后续 fetch 不用再拼参数
                extra['Set-Cookie'] = 'gp_tok=%s; Path=/; SameSite=Lax' % TOKEN
            self._send(200, 'text/html; charset=utf-8', render_page(), extra)
        elif path == '/api/state':
            with SESSION.lock:
                self._json(SESSION.snapshot())
        elif path == '/api/models':
            self._json({'models': list_models()})
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        if not self._token_ok():
            self._json({'error': '需要访问令牌'}, 401)
            return
        path = self.path.split('?')[0]
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            body = {}
        try:
            with SESSION.lock:
                if path == '/api/new':
                    self._json(SESSION.reset(body.get('model'),
                                            body.get('playout'),
                                            body.get('human_first')))
                elif path == '/api/move':
                    self._json(SESSION.human_move(int(body.get('move'))))
                elif path == '/api/ai':
                    self._json(SESSION.ai_move())
                elif path == '/api/undo':
                    self._json(SESSION.undo())
                elif path == '/api/hint':
                    self._json(SESSION.make_hint())
                else:
                    self._json({'error': 'not found'}, 404)
        except (ValueError, TypeError) as exc:
            self._json({'error': str(exc)}, 400)
        except Exception as exc:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._json({'error': '%s: %s' % (type(exc).__name__, exc)}, 500)


def free_port(host, want):
    """找一个能绑的端口。

    探测 socket 必须设 SO_REUSEADDR：上一次运行的 keep-alive 连接会留在
    CLOSE-WAIT/FIN-WAIT-2，不设这项就绑不上，会把服务无声挪到下一个端口，
    而 HTTPServer 自己设了 allow_reuse_address，本来是能绑的。
    """
    for p in [want] + list(range(want + 1, want + 20)):
        s = socket.socket()
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    raise SystemExit('找不到空闲端口')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1',
                    help='0.0.0.0 表示允许同一局域网内的其他设备访问')
    ap.add_argument('--port', type=int, default=8777)
    ap.add_argument('--model', default=None)
    ap.add_argument('--playout', type=int, default=400)
    ap.add_argument('--c-puct', type=float, default=5.0)
    ap.add_argument('--first', choices=('human', 'ai'), default='human')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--no-token', action='store_true',
                    help='对外监听时不加访问令牌（不推荐）')
    a = ap.parse_args()

    # 只跑 Theano pickle 模型时不需要 torch；有就顺手把线程数压到 1
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    SESSION.c_puct = a.c_puct
    models = list_models()
    if not models:
        raise SystemExit('目录里没有 .model 文件')
    model = a.model or models[0]

    SESSION.reset(model=model, playout=a.playout,
                  human_first=(a.first == 'human'))
    print('模型 %s | %s | %dx%d / %d 连' % (
        model, SESSION.brain(model).backend, SESSION.side, SESSION.side,
        SESSION.n_in_row))

    shared = a.host not in ('127.0.0.1', 'localhost', '::1')
    global TOKEN
    if shared and not a.no_token:
        TOKEN = secrets.token_urlsafe(6)

    port = free_port(a.host, a.port)
    srv = ThreadingHTTPServer((a.host, port), Handler)
    srv.daemon_threads = True

    local = 'http://127.0.0.1:%d/' % port
    print('本机打开   %s' % local)
    if shared:
        tail = ('/?t=%s' % TOKEN) if TOKEN else '/'
        print('局域网访问（手机/平板要和这台机器在同一个 WiFi 或热点下）:')
        addrs = lan_addresses()
        for name, ip in addrs:
            print('   %-8s http://%s:%d%s' % (name, ip, port, tail))
        if not addrs:
            print('   没找到局域网网卡，用 `ip addr` 自己看一眼。')
        if TOKEN:
            print('   链接里带访问令牌，别人拿到同样能玩；Ctrl-C 结束即失效。')
        else:
            print('   ⚠️ 已关闭令牌，同网段任何人都能打开。')
    sys.stdout.flush()

    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(local)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')


if __name__ == '__main__':
    main()
