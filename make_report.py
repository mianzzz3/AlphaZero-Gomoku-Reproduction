# -*- coding: utf-8 -*-
"""从训练日志与指标 CSV 生成自包含的可视化报告。

用法：
    python make_report.py

输出：training_report.html（单文件，数据内嵌，直接双击打开）
"""

from __future__ import print_function

import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_DIR = os.path.join(HERE, 'gpu_run_20261113_104802')

METRIC_RE = re.compile(
    r'kl:([\d.eE+-]+),lr_multiplier:([\d.eE+-]+),loss:([\d.eE+-]+),'
    r'entropy:([\d.eE+-]+),explained_var_old:([\d.eE+-]+),'
    r'explained_var_new:([\d.eE+-]+),policy_loss:([\d.eE+-]+),'
    r'value_loss:([\d.eE+-]+)')
BATCH_RE = re.compile(r'batch i:(\d+), episode_len:(\d+)')
EVAL_HEAD_RE = re.compile(r'current self-play batch: (\d+)')
EVAL_RE = re.compile(r'num_playouts:(\d+), win: (\d+), lose: (\d+), tie:(\d+)')


def parse_gpu_log(path):
    """返回 (metrics, episodes, evals)。metrics 每条对应 batch = 序号 + 1。"""
    metrics, episodes, evals = [], [], []
    pending_batch = None
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = BATCH_RE.search(line)
            if m:
                episodes.append([int(m.group(1)), int(m.group(2))])
                continue
            m = EVAL_HEAD_RE.search(line)
            if m:
                pending_batch = int(m.group(1))
                continue
            m = EVAL_RE.search(line)
            if m:
                evals.append([pending_batch, int(m.group(1)),
                              int(m.group(2)), int(m.group(3)), int(m.group(4))])
                pending_batch = None
                continue
            m = METRIC_RE.search(line)
            if m:
                metrics.append([float(x) for x in m.groups()])
    for i, row in enumerate(metrics):
        row.insert(0, i + 1)
    return metrics, episodes, evals


def parse_cpu_csv(metric_path, eval_path):
    metrics = []
    with open(metric_path) as fh:
        for row in csv.DictReader(fh):
            metrics.append([
                int(row['batch']),
                float(row['kl']),
                float(row['lr_multiplier']),
                float(row['loss']),
                float(row['entropy']),
                float(row['explained_var_old']),
                float(row['explained_var_new']),
            ])
    evals = []
    with open(eval_path) as fh:
        for row in csv.DictReader(fh):
            evals.append([int(row['batch']), int(row['pure_mcts_playouts']),
                          int(row['wins']), int(row['losses']), int(row['ties'])])
    return metrics, evals


def downsample(rows, step):
    if step <= 1:
        return rows
    out = rows[::step]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def moving_avg(values, window):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def build():
    gpu_metrics, gpu_eps, gpu_evals = parse_gpu_log(
        os.path.join(GPU_DIR, 'train_output.log'))
    cpu_metrics, cpu_evals = parse_cpu_csv(
        os.path.join(HERE, 'train_output_metrics.csv'),
        os.path.join(HERE, 'train_output_evaluations.csv'))

    # 字段顺序: batch, kl, lr_mult, loss, entropy, ev_old, ev_new, policy_loss, value_loss
    step = 3
    g = downsample(gpu_metrics, step)
    data = {
        'gpu': {
            'batch': [r[0] for r in g],
            'kl': [r[1] for r in g],
            'lr': [r[2] for r in g],
            'loss': [r[3] for r in g],
            'entropy': [r[4] for r in g],
            'ev_old': [r[5] for r in g],
            'ev_new': [r[6] for r in g],
            'policy_loss': [r[7] for r in g],
            'value_loss': [r[8] for r in g],
        },
        'gpu_ep': {
            'batch': [r[0] for r in gpu_eps],
            'len': [r[1] for r in gpu_eps],
        },
        'gpu_eval': gpu_evals,
        'cpu': {
            'batch': [r[0] for r in cpu_metrics],
            'kl': [r[1] for r in cpu_metrics],
            'lr': [r[2] for r in cpu_metrics],
            'loss': [r[3] for r in cpu_metrics],
            'entropy': [r[4] for r in cpu_metrics],
            'ev_new': [r[6] for r in cpu_metrics],
        },
        'cpu_eval': cpu_evals,
    }

    # 关键统计
    total = len(gpu_metrics)
    # 「仍具备解释力」的最后一个 batch：|evn| > 0.01。此后只剩 ±0.007 的浮点噪声。
    last_meaningful = 0
    tail_max = 0.0
    for r in gpu_metrics:
        if abs(r[6]) > 0.01:
            last_meaningful = r[0]
    for r in gpu_metrics:
        if r[0] > last_meaningful:
            tail_max = max(tail_max, abs(r[6]))
    stats = {
        'games': gpu_eps[-1][0] if gpu_eps else 0,
        'updates': total,
        'ev_last_nonzero': last_meaningful,
        'ev_tail_max': round(tail_max, 4),
        'ev_zero_ratio': round(
            sum(1 for r in gpu_metrics if r[6] == 0.0) / total, 4),
        'final_entropy': round(gpu_metrics[-1][4], 2),
        'final_policy_loss': round(gpu_metrics[-1][7], 2),
        'final_value_loss': round(gpu_metrics[-1][8], 3),
        'final_ep_len': gpu_eps[-1][1] if gpu_eps else 0,
        'min_ep_len': min(r[1] for r in gpu_eps) if gpu_eps else 0,
        'cpu_lr_final': round(cpu_metrics[-1][2], 2),
        'cpu_lr_real': round(2e-3 * cpu_metrics[-1][2], 4),
        'gpu_lr_final': round(gpu_metrics[-1][2], 3),
        'gpu_lr_real': round(2e-3 * gpu_metrics[-1][2], 5),
        'evals': len(gpu_evals),
    }

    # 评估：胜率序列
    data['gpu_eval_ratio'] = [
        [e[0], e[1], round((e[2] + 0.5 * e[4]) / max(1, e[2] + e[3] + e[4]), 2)]
        for e in gpu_evals]

    html = TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__STATS__', json.dumps(stats, separators=(',', ':')))

    # 内联 Chart.js，保证离线双击也能渲染；缺失时退回 CDN。
    vendor = os.path.join(HERE, 'vendor', 'chart.umd.js')
    if os.path.exists(vendor):
        with open(vendor, encoding='utf-8') as fh:
            chart_src = fh.read()
        html = html.replace('__CHARTJS__',
                            '<script>\n' + chart_src + '\n</script>')
    else:
        html = html.replace(
            '__CHARTJS__',
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/'
            'Chart.js/4.4.1/chart.umd.js"></script>')

    out_path = os.path.join(HERE, 'training_report.html')
    with open(out_path, 'w') as fh:
        fh.write(html)
    print('written:', out_path)
    print('gpu updates:', total, '| games:', stats['games'],
          '| ev_new last meaningful batch:', last_meaningful,
          '| tail max:', stats['ev_tail_max'])
    return out_path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaZero-Gomoku 训练复盘</title>
__CHARTJS__
<style>
:root{
  --bg:#141414; --card:#1c1c1c; --line:rgba(255,255,255,.09);
  --fg:#e8e8e6; --dim:#9a9a96; --dim2:#6f6f6b;
  --coral:#F0997B; --amber:#EF9F27; --blue:#85B7EB; --teal:#5DCAA5;
  --red:#F09595; --gray:#888780; --purple:#AFA9EC;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:"Noto Sans CJK SC","Source Han Sans CN","PingFang SC",system-ui,sans-serif;
  font-size:14px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 72px}
h1{font-size:26px;font-weight:500;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin:0 0 32px}
h2{font-size:15px;font-weight:500;margin:0 0 4px}
.cap{color:var(--dim2);font-size:12px;margin:0 0 14px}
section{margin-bottom:40px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:36px}
.card{background:var(--card);border:.5px solid var(--line);border-radius:12px;padding:16px 18px}
.card .k{color:var(--dim);font-size:12px;margin-bottom:6px}
.card .v{font-size:24px;font-weight:500;letter-spacing:-.02em}
.card .n{color:var(--dim2);font-size:12px;margin-top:4px}
.verdict{background:var(--card);border:.5px solid var(--line);border-left:2px solid var(--red);
  border-radius:12px;padding:18px 20px;margin-bottom:36px}
.verdict p{margin:0 0 10px}
.verdict p:last-child{margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:20px}
.panel{background:var(--card);border:.5px solid var(--line);border-radius:12px;padding:18px 18px 12px}
.chart{position:relative;height:260px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--dim);margin-bottom:10px}
.legend i{display:inline-block;width:13px;height:2px;vertical-align:middle;margin-right:5px;border-radius:1px}
.legend i.dash{border-top:1px dashed;height:0}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line)}
th{color:var(--dim);font-weight:400;font-size:12px}
td.num{font-variant-numeric:tabular-nums}
code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;
  background:rgba(255,255,255,.06);padding:1px 5px;border-radius:4px}
.note{color:var(--dim);font-size:12.5px;margin-top:10px;border-top:.5px solid var(--line);padding-top:10px}
</style>
</head>
<body>
<div class="wrap">
  <h1>AlphaZero-Gomoku 训练复盘</h1>
  <p class="sub">10×10 棋盘 · 连五 · 400 MCTS playouts · 数据源：gpu_run_20261113_104802/train_output.log</p>

  <div class="cards" id="cards"></div>

  <div class="verdict">
    <p><strong>结论：这轮训练没有产出有效模型。</strong></p>
    <p>价值网络在第 <span id="v_batch"></span> 局之后再没产生过有效信号：此后 <code>explained_var</code> 的最大绝对值只有 <span id="v_tail"></span>，<span id="v_zero"></span> 的更新记录精确为 0；<code>value_loss</code> 锁死在 1.000 —— 那正是「预测恒等于标签均值」的 MSE 下界。MCTS 的叶节点评估依赖价值网络，价值函数失效意味着搜索退化为「先验 + 访问计数」，策略梯度只能拟合一个已经退化了的搜索分布。</p>
    <p>同期自对弈步数从 39 掉到 18，最短 9 手 —— 先手 5 手连五、后手全程不挡。网络学到的是先手背诵脚本，不是棋力。</p>
  </div>

  <section>
    <h2>损失分解</h2>
    <p class="cap">policy_loss 在收敛，value_loss 从 0.46 一路爬回 1.000 —— 后者正是「预测恒等于标签均值」的 MSE 下界。</p>
    <div class="panel">
      <div class="legend">
        <span><i style="background:var(--blue)"></i>policy_loss</span>
        <span><i style="background:var(--red)"></i>value_loss</span>
        <span><i class="dash" style="border-color:var(--gray)"></i>总损失</span>
      </div>
      <div class="chart"><canvas id="loss" role="img" aria-label="策略损失下降、价值损失回到 1.0 的折线图">policy_loss 从 4.6 降到 2.3，value_loss 从 0.46 升到 1.00。</canvas></div>
    </div>
  </section>

  <section>
    <h2>价值网络的有效性</h2>
    <p class="cap">explained_var 只在最初 24 局有非零值（峰值 0.778 出现在第 2 个 batch），之后精确归零。右侧放大前 300 局。</p>
    <div class="grid">
      <div class="panel">
        <div class="legend"><span><i style="background:var(--amber)"></i>explained_var (更新后)</span></div>
        <div class="chart"><canvas id="evfull" role="img" aria-label="explained_var 全程折线图，第 24 局后归零">explained_var 在第 24 个 batch 后恒为 0。</canvas></div>
      </div>
      <div class="panel">
        <div class="legend"><span><i style="background:var(--amber)"></i>explained_var (前 300 局)</span></div>
        <div class="chart"><canvas id="evzoom" role="img" aria-label="前 300 局 explained_var 放大图">前 300 局内 explained_var 有几次非零峰值后归零。</canvas></div>
      </div>
    </div>
  </section>

  <section>
    <h2>自对弈退化</h2>
    <p class="cap">对局步数与策略熵同步下滑。熵的均匀分布基准是 ln(100) = 4.605。</p>
    <div class="panel">
      <div class="legend">
        <span><i style="background:var(--coral)"></i>自对弈步数</span>
        <span><i class="dash" style="border-color:var(--blue)"></i>策略熵</span>
      </div>
      <div class="chart" style="height:290px"><canvas id="degen" role="img" aria-label="自对弈步数与策略熵双轴折线图">自对弈步数由 39 降至 18，策略熵由 4.4 降至 2.2。</canvas></div>
    </div>
  </section>

  <section>
    <h2>评估与对手预算</h2>
    <p class="cap">每 50 局评 10 盘，先手轮换。对手搜索预算随胜率上调，因此柱高与预算必须一起看。</p>
    <div class="panel">
      <div class="legend">
        <span><i style="background:var(--teal)"></i>对纯 MCTS 胜率（10 局）</span>
        <span><i class="dash" style="border-color:var(--purple)"></i>对手 playouts（千）</span>
      </div>
      <div class="chart" style="height:280px"><canvas id="evalc" role="img" aria-label="评估胜率与对手预算随训练变化">胜率从 0 逐步升到 9/10，对手预算同期从 1000 提到 5000。</canvas></div>
    </div>
  </section>

  <section>
    <h2>学习率：两次训练朝两个方向跑偏</h2>
    <p class="cap">同一套 KL 自适应逻辑，GPU 那轮被一路 /1.5 压到 0.088 倍，CPU 那轮被一路 ×1.5 顶到 11.4 倍（实际 lr = 0.0228）。</p>
    <div class="panel">
      <div class="legend">
        <span><i style="background:var(--teal)"></i>GPU 训练 lr_multiplier</span>
        <span><i style="background:var(--amber)"></i>CPU 训练 lr_multiplier</span>
      </div>
      <div class="chart" style="height:260px"><canvas id="lr" role="img" aria-label="两次训练的 lr_multiplier 对比">GPU 学习的 lr_multiplier 降到 0.088，CPU 升到 11.39。</canvas></div>
    </div>
    <p class="note" id="lrnote"></p>
  </section>

  <section>
    <h2>关键数字</h2>
    <div class="panel"><table id="tbl"></table></div>
  </section>
</div>

<script>
const D = __DATA__;
const S = __STATS__;
const C = {coral:'#F0997B',amber:'#EF9F27',blue:'#85B7EB',teal:'#5DCAA5',
           red:'#F09595',gray:'#888780',purple:'#AFA9EC'};
const GRID = 'rgba(255,255,255,.06)', TICK = '#8f8f8b';
Chart.defaults.font.family = '"Noto Sans CJK SC","Source Han Sans CN",system-ui,sans-serif';
Chart.defaults.font.size = 11;
Chart.defaults.color = TICK;

const base = {
  responsive:true, maintainAspectRatio:false, animation:false,
  interaction:{mode:'index',intersect:false},
  elements:{point:{radius:0,hoverRadius:3}},
  plugins:{legend:{display:false},
    tooltip:{backgroundColor:'#262625',titleColor:'#e8e8e6',bodyColor:'#c8c8c4',
      borderColor:'rgba(255,255,255,.12)',borderWidth:1,padding:10,displayColors:true}}
};
const ax = (title, color) => ({
  grid:{color:GRID,drawTicks:false}, border:{display:false},
  ticks:{color:color||TICK, maxTicksLimit:6},
  title:{display:!!title, text:title, color:color||TICK, font:{size:11}}
});

document.getElementById('cards').innerHTML = [
  ['自对弈局数', S.games, '目标 3000，在第 '+S.games+' 局中断'],
  ['网络更新次数', S.updates, '每次更新 5 个 epoch × batch 512'],
  ['价值头存活', S.ev_last_nonzero+' 局', '此后 explained_var 只剩 ±'+S.ev_tail_max],
  ['最终策略熵', S.final_entropy, '均匀分布 = 4.605'],
].map(([k,v,n]) => `<div class="card"><div class="k">${k}</div>
  <div class="v">${v}</div><div class="n">${n}</div></div>`).join('');

document.getElementById('v_batch').textContent = S.ev_last_nonzero;
document.getElementById('v_tail').textContent = S.ev_tail_max;
document.getElementById('v_zero').textContent = (S.ev_zero_ratio*100).toFixed(1)+'%';

new Chart(document.getElementById('loss'), {
  type:'line',
  data:{labels:D.gpu.batch,datasets:[
    {label:'policy_loss',data:D.gpu.policy_loss,borderColor:C.blue,borderWidth:1.8,tension:.2},
    {label:'value_loss',data:D.gpu.value_loss,borderColor:C.red,borderWidth:1.8,tension:.2},
    {label:'loss',data:D.gpu.loss,borderColor:C.gray,borderWidth:1,borderDash:[4,3],tension:.2}
  ]},
  options:{...base,scales:{x:ax('自对弈局数'),y:ax('损失',null)}}
});

new Chart(document.getElementById('evfull'), {
  type:'line',
  data:{labels:D.gpu.batch,datasets:[
    {label:'explained_var',data:D.gpu.ev_new,borderColor:C.amber,borderWidth:1.4,tension:0}
  ]},
  options:{...base,scales:{x:ax('自对弈局数'),y:{...ax('explained_var',C.amber),min:-0.1,max:0.85}}}
});

const zi = D.gpu.batch.findIndex(b => b > 300);
new Chart(document.getElementById('evzoom'), {
  type:'line',
  data:{labels:D.gpu.batch.slice(0,zi),datasets:[
    {label:'explained_var',data:D.gpu.ev_new.slice(0,zi),borderColor:C.amber,
      borderWidth:1.6,pointRadius:1.6,pointBackgroundColor:C.amber,tension:0}
  ]},
  options:{...base,scales:{x:ax('自对弈局数'),y:{...ax('explained_var',C.amber),min:-0.1,max:0.85}}}
});

new Chart(document.getElementById('degen'), {
  type:'line',
  data:{labels:D.gpu_ep.batch,datasets:[
    {label:'自对弈步数',data:D.gpu_ep.len,borderColor:C.coral,borderWidth:1.2,yAxisID:'y',tension:.15},
    {label:'策略熵',data:D.gpu.entropy,borderColor:C.blue,borderWidth:1.6,borderDash:[5,3],yAxisID:'y1',tension:.15}
  ]},
  options:{...base,scales:{
    x:ax('自对弈局数'),
    y:{...ax('步数',C.coral),min:5,max:70},
    y1:{...ax('策略熵',C.blue),min:2,max:4.8,position:'right',grid:{drawOnChartArea:false},border:{display:false}}
  }}
});

new Chart(document.getElementById('evalc'), {
  type:'bar',
  data:{labels:D.gpu_eval_ratio.map(r=>r[0]),datasets:[
    {label:'胜率',data:D.gpu_eval_ratio.map(r=>r[2]),backgroundColor:'rgba(93,202,165,.55)',
      borderColor:C.teal,borderWidth:1,yAxisID:'y',barPercentage:.6},
    {label:'对手 playouts',data:D.gpu_eval_ratio.map(r=>r[1]/1000),borderColor:C.purple,
      borderWidth:1.6,borderDash:[4,3],type:'line',yAxisID:'y1',tension:0,pointRadius:2,pointBackgroundColor:C.purple}
  ]},
  options:{...base,scales:{
    x:{...ax('自对弈局数'),ticks:{color:TICK,maxTicksLimit:10,maxRotation:45}},
    y:{...ax('胜率',C.teal),min:0,max:1.05,ticks:{color:C.teal,callback:v=>v.toFixed(1)}},
    y1:{...ax('对手 playouts（千）',C.purple),min:0,max:5.5,position:'right',grid:{drawOnChartArea:false},border:{display:false}}
  }}
});

new Chart(document.getElementById('lr'), {
  type:'line',
  data:{datasets:[
    {label:'GPU',data:D.gpu.batch.map((b,i)=>({x:b,y:D.gpu.lr[i]})),
      borderColor:C.teal,borderWidth:1.8,tension:.1},
    {label:'CPU',data:D.cpu.batch.map((b,i)=>({x:b,y:D.cpu.lr[i]})),
      borderColor:C.amber,borderWidth:1.8,tension:.1}
  ]},
  options:{...base,scales:{
    x:{...ax('自对弈局数'),type:'linear',min:0,max:3100},
    y:{...ax('lr_multiplier'),type:'logarithmic',min:0.05,max:20,
      ticks:{color:TICK,callback:v=>v}}
  }}
});

document.getElementById('lrnote').innerHTML =
  `CPU 训练停在 <code>lr_multiplier = ${S.cpu_lr_final}</code>，实际学习率 <code>${S.cpu_lr_real}</code>；`
  + `GPU 训练停在 <code>${S.gpu_lr_final}</code>，实际学习率 <code>${S.gpu_lr_real}</code>。`
  + ' 代码里 <code>elif kl &lt; kl_targ/2 and lr_multiplier &lt; 10</code> 的判断放在乘法之前，'
  + '上限实际是 15 而非 10；而 CPU 那轮的 KL 只有 0.0004，远低于阈值，于是学习率被一路放大。';

document.getElementById('tbl').innerHTML =
  '<tr><th>指标</th><th>数值</th><th>含义</th></tr>' +
  [['最终 episode_len', S.final_ep_len, '最短曾到 '+S.min_ep_len+' 手'],
   ['最终 entropy', S.final_entropy, '均匀分布 4.605'],
   ['最终 policy_loss', S.final_policy_loss, '均匀分布目标约 4.6'],
   ['最终 value_loss', S.final_value_loss, '常数零预测的理论下界 = 1.0'],
   ['explained_var 为零的比例', (S.ev_zero_ratio*100).toFixed(1)+'%', '全程 '+S.updates+' 次更新'],
   ['评估次数', S.evals, '每 50 局一次，每次 10 盘'],
   ['CPU 训练 lr_multiplier', S.cpu_lr_final, '实际 lr '+S.cpu_lr_real],
  ].map(([a,b,c])=>`<tr><td>${a}</td><td class="num">${b}</td><td style="color:var(--dim)">${c}</td></tr>`).join('');
</script>
</body>
</html>
"""


if __name__ == '__main__':
    build()
