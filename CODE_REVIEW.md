# 代码审查：AlphaZero-Gomoku 复现

审查范围：仓库全部 Python 源码 + `gpu_run_20261113_104802/` 与根目录的训练日志、指标、模型文件。

---

## 结论速览

1. **价值网络彻底失效，但策略网络是能用的 —— 这是半成品，不是废品。** 价值头在第 24 个 batch 之后输出恒为常数（`explained_var` 精确等于 `0.000`，`value_loss` 恒等于 `1.000`），此后 2370 局都在训练一个死掉的价值头。`RESULTS.md` 里「对 256 个随机合法局面的价值预测均约为 0.0225」不是结果，是失效证据。
   但 `policy_loss` 从 4.60 降到 2.29、`entropy` 从 3.85 降到 2.31，策略确实学到了东西。实测 `best_policy.model` 对同棋盘 200 模拟/步的纯 MCTS 打 6 局是 **4 胜 2 负**，而完全没训的随机策略会全负。代价是 MCTS 的叶子评估恒为 0，搜索退化成「先验 + 访问计数」，不会判断形势，只能靠走子直觉。
2. **代码能跑通，问题集中在三处**：`game.py` 的坐标与颜色通道语义、`train.py` 的 KL 自适应和评估噪声、`mcts_alphaZero.py` 的效率。算法主干（PUCT、递归回传、对称增强）是对的。
3. **代码隐含两个未写出来的前提**：棋盘必须是正方形；自对弈必须永远 player1 先手。前者不满足会静默算错，后者会让颜色通道与「我是 player1」完全共线。
4. **能直接拿来下棋的是上游模型**，不是自己训的这份。按 6 局对同棋盘纯 MCTS 的得分：`best_policy_8_8_5` 6 胜 0 负、`best_policy.model` 4 胜 2 负、`best_policy_6_6_4` 6 胜 0 负、60 局修复版 0 胜 6 负。
5. **现在可以图形化下棋**：`python web_play.py`，浏览器打开 `http://127.0.0.1:8777`。界面会把「AI 自评胜率、候选分布、搜索覆盖了多少点」直接画出来 —— 价值头死没死，不用看曲线，下几手就看得出来。

---

## 一、训练失效的量化证据

按自对弈局数分段统计 GPU 日志（2398 条更新记录）：

| 局数区间 | episode_len 均值 | entropy | policy_loss | value_loss | explained_var |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0–200 | 36.2 | 3.85 | 3.84 | 1.04 | -0.019 |
| 200–600 | 24.8 | 2.97 | 2.95 | 1.00 | 0.000 |
| 600–1200 | 18.6 | 2.64 | 2.61 | 1.00 | 0.000 |
| 1200–1800 | 17.8 | 2.45 | 2.44 | 1.00 | 0.000 |
| 1800–2399 | 18.2 | 2.31 | 2.29 | 1.00 | 0.000 |

三条独立的失效信号：

- `explained_var` 只在**前 24 个 batch** 有非零值（最大 0.778，出现在第 2 个 batch），第 1977 个 batch 之后彻底不再出现非零；`value_loss` 精确锁死在 1.000。价值头的预测等于 batch 内 `z` 的均值，也等于常数 0 —— 因为 `z` 按走子方视角取 ±1、先手后手各占一半，均值就是 0。此时 MSE 的理论下界正好是 1.0。
- `episode_len` 从 36 掉到 18，最小值 9。9 手结束意味着先手第 5 手连五、后手全程不挡。**自对弈退化成「先手背诵胜利脚本、后手放弃防守」**。
- `entropy` 从 3.85 跌到 2.31（10×10 均匀分布熵为 ln 100 = 4.61）。策略确实在收敛，但收敛到的是上述退化脚本，不是棋力。

价值头失效的机制（按可能性排序，未做训练验证）：

1. **共享卷积层被策略梯度占满**。`conv1-3` 由两个头共用，policy 梯度量级显著大于 value（`(p-π)` 在 100 个动作上求和后除以 batch），value 分支只剩 `val_conv1 → fc1 → fc2` 三段能学，而它们的输入特征本身没被优化。
2. **学习率被 KL 机制压到 1.76e-4**（末段 `lr_multiplier = 0.088`），此时让 `tanh` 饱和到 ±1 所需的权重增长在 2200 局内根本积累不起来。
3. **`z` 的均值≈0 让常数解成为强吸引子**。`v=0` 时梯度只有 `2z/N ≈ 0.004`，与 `weight_decay=1e-4` 的收缩力同量级。

验证方法（改代码时加 3 行即可判定）：训练时打印 `v.std()`、`(v*z).mean()/z.std()`、`val_fc2.weight.norm()`。若 `v.std()` 趋 0 且 `fc2.weight.norm()` 单调下降 → 是第 3 条；若 `fc2` 范数正常但 `v.std()` 仍趋 0 → 是第 1 条。

### 三个 checkpoint 的实测

在 40 个随机构造的局面（合法落子 0–25 手、随机先手方）上直接跑前向：

| checkpoint | v 均值 | v 标准差 | 取值范围 |
| --- | ---: | ---: | --- |
| `gpu_run_*/best_policy.model` | +0.0225 | 0.0000 | [+0.022, +0.022] |
| `gpu_run_*/current_policy.model` | +0.0225 | 0.0000 | [+0.022, +0.022] |
| `best_policy.model`（CPU 那轮） | +0.0567 | 0.0000 | [+0.057, +0.057] |

两轮训练、三个 checkpoint，价值输出在任意局面上完全相同。`RESULTS.md` 里「对 256 个随机合法局面的价值预测均约为 0.0225」这句，字面就是这个现象的全部——不是收敛得太平滑，是根本没有输入依赖。

自对弈一局（10×10、每步 400 次模拟，见 `demo_game.html`）的逐步记录：19 手内网络价值恒为 `+0.057`；除最后一手直接连五（置信度 0.89）外，其余每步搜索后的首选点访问占比都在 0.02–0.03，即 400 次模拟摊到 100 个空位、每点 4 次。作为对照，上游 8×8 模型在同样 400 次模拟下，价值全程在 [-0.998, +1.000] 之间变化，首选点访问占比中位数 0.39。


---

## 二、会直接报错的问题

### 1. `human_play.py` 加载不了你自己训练的模型

> **已修复。** 现在按文件头自动识别 PyTorch / Theano 格式，棋盘尺寸从权重推断，默认优先加载修复后的模型。见第八节。

`human_play.py:52` 默认 `model_file = 'best_policy_8_8_5.model'`，走 `pickle.load` → `PolicyValueNetNumpy`。这条路径只对**上游 Theano 训练出的 pickle 模型**成立（`best_policy_8_8_5.model`、`best_policy_6_6_4.model` 的文件头是 `80 02`，即 pickle protocol 2）。

你自己训练的 `best_policy.model` / `current_policy.model` 文件头是 `50 4b 03 04`（zip），是 `torch.save(state_dict)` 格式。`pickle.load` 对它抛 `UnpicklingError`；`encoding='bytes'` 的 fallback 也救不了。而且 numpy 版 `conv_forward` 专门按 Theano 的卷积核翻转写的（`policy_value_net_numpy.py:29`），**即使能读出权重也会算错**。

想跟自己的 10×10 模型下棋，必须换成：

```python
from policy_value_net_pytorch import PolicyValueNet
board = Board(width=10, height=10, n_in_row=5)
best_policy = PolicyValueNet(10, 10, model_file='best_policy.model', use_gpu=False)
```

### 2. `train.py` 硬编码 `use_gpu=True`，没有可用性检查

`train.py:55` 和 `train.py:60` 两处直接传 `use_gpu=True`，`policy_value_net_pytorch.py:70` 无条件 `.cuda()`。在无 GPU 机器上 `import` 完就崩，报错位置离真正原因很远。`RESULTS.md` 里那条启动命令带了 `assert torch.cuda.is_available()`，算是外部兜底，但脚本本身应该自己检查。

### 3. `torch.load` 没有 `map_location`

`policy_value_net_pytorch.py:77`：`net_params = torch.load(model_file)`。在 GPU 上保存的 checkpoint 拿到 CPU 机器上加载会失败。所有 `model_file=` 的加载路径都有这个问题（包括 `--init_model` 续训）。改成 `torch.load(model_file, map_location='cpu' if not self.use_gpu else None)` 即可。

已实测复现：`gpu_run_20261113_104802/` 下的两个模型是在 GPU 上保存的，在纯 CPU 环境里 `PolicyValueNet(..., model_file=...)` 直接抛
`RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`。
根目录那两个（CPU 那轮保存）能加载，所以这个 bug 只在换机器时才暴露。

---

## 三、静默算错的问题（不报错，结果错）

### 1. 棋盘非正方形时坐标全错 —— `game.py:66-72`

```python
square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
```

`% self.height` 应该是 `% self.width`。四个通道、`last_move` 一共 4 处同样的错。实测 4×6 棋盘下 `move=9`（应为 h=2,w=1）被写到 (1,3)；6×4 棋盘下直接写越界区域。

因为所有模型都是正方形，这个 bug 从未暴露。但 `Board(width, height)` 的签名宣称支持非方棋盘 —— 要么修好，要么显式 `assert width == height`，别留一个假的自由度。`train.py:76` 的 `get_equi_data` 里 `mcts_prob.reshape(self.board_height, self.board_width)` 也是同一个前提。

### 2. 颜色通道的语义在 `start_player=1` 时反转 —— `game.py:73-74`

```python
if len(self.states) % 2 == 0:
    square_state[3][:, :] = 1.0
```

这里是按**步数奇偶**推「当前是不是先手方」，代码注释写的是「indicate the colour to play」。训练时 `start_self_play` 永远 `init_board()`（默认 `start_player=0`），所以通道 3 ≡ 1 当且仅当当前玩家是 player1，两者恰好等价，训练不受影响。

但 `train.py:164` 的 `policy_evaluate` 用 `start_player=i % 2` 轮换先手。i 为奇数时 player2 先手，第一步 `len(states)=0` → 通道 3 = 1，语义却变成了「我是 player2」。**评估时网络收到的是训练时从未见过的特征分布**，先手轮换那 5 局的结果因此系统性偏移。

正确写法：

```python
square_state[3][:, :] = 1.0 if self.current_player == self.players[0] else 0.0
```

### 3. `explained_var` 会除零 —— `train.py:126-131`

```python
np.var(np.array(winner_batch))
```

若某个 mini-batch 的 `z` 全相同（自对弈退化到「先手必胜」时完全可能），方差为 0 → `inf`/`nan`。当前日志里没触发，但在退化数据上迟早会。加个 `+ 1e-8` 或者显式跳过。

### 4. `lr_multiplier` 的上限判断写错 —— `train.py:124`

```python
elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
    self.lr_multiplier *= 1.5
```

判断放在乘法**之前**，所以乘完后可以越过 10。CPU 那轮训练的 `lr_multiplier` 精确停在 11.391（1.5 的幂次序列 7.59 → 11.39，下一次因 11.39 > 10 不再乘），实际学习率 = `2e-3 × 11.391 = 0.0228`，是初始值的 11 倍。要卡上限应该写成 `self.lr_multiplier = min(self.lr_multiplier * 1.5, 10)`。

---

## 四、让训练退化的设计缺陷（不是 bug，但决定了结果）

### 1. 自对弈从不换边 —— `game.py:194`

`start_self_play` 调用的是 `self.board.init_board()`，没有传 `start_player`。所以 2399 局里**先手永远是 player1**。后果：

- 通道 3 与「我是 player1」完全共线，网络学不到「先手/后手」这个真正重要的信息，只能硬编码「player1 优先」。
- 数据里没有「同一局面换边后该怎么做」的对照，价值函数无法区分「我领先」和「轮到我了」。
- 退化成先手必胜脚本后，`z` 的分布彻底失去方差，价值头的常数解变成最优解（参见第一节）。

修法：`init_board(np.random.randint(2))`。一行。

### 2. 温度全程 1.0，从不降温 —— `train.py:35` + `game.py:190`

`self.temp = 1.0` 一路传给 `start_self_play`，整个对局都用 `temp=1.0` 计算 π。AlphaZero 的做法是前若干步 `temp=1`、之后趋近 0。当前配置下：

- π 始终是访问数的**比例**，高熵目标。entropy 停在 2.31 有一半是这个原因。
- 加上 10×10 的对局只有 18 手，所谓「开局探索期」覆盖了整局，等于从不降温。
- 落子还用 `0.75π + 0.25 Dirichlet` 采样，残局继续随机走，训练数据的走子质量无法随训练提升。

### 3. KL 自适应在 batch 512 + 5 epoch 的配置下完全失效

`kl_targ = 0.02`，但实测 KL 长期在 0.023–0.027（GPU）和 0.0004（CPU）。原因：一次 update 只做 5 个 epoch，KL 天然就小，这个阈值是按上游 6×6 小棋盘的配置定的。

- CPU 那轮：KL 0.0004 远低于 `kl_targ/2 = 0.01` → 每步 ×1.5 → lr 涨到 11.4 倍 → 0.0228。这是 CPU 版 loss 一直在 5.2–6.5、entropy 4.2 的直接原因。
- GPU 那轮：KL 略高于 0.02 → lr 一路被 /1.5 压到 0.088 倍。价值头在低 lr 下更学不动。

同一套逻辑在两个方向上都跑偏了。要么把 `kl_targ` 改成和配置匹配（10×10 + batch 512 + 5 epoch 下应显著大于 0.02），要么干脆去掉自适应、固定 lr。

### 4. 400 playout 对 10×10 太少

`n_playout=400` 是上游给 6×6 的建议值。10×10 有 100 个合法点，400 次模拟等于每点平均访问 4 次，PUCT 的 Q 估计噪声极大。`README.md` 自己也写了 8×8 要 2000–3000 局，你直接上 10×10。这一条和 §1 的价值塌缩是互相放大的：v 越不准，MCTS 越退化；MCTS 越退化，π 越差；π 越差，v 越学不到。

### 5. 评估只用 10 局，`best_win_ratio` 严重过拟合噪声

`train.py:150` `policy_evaluate(n_games=10)`。CPU 那轮 10 次评估的胜率是 0.2 / 0.0 / 0.4 / 0.4 / 0.3 / 0.1 / 0.4 / 0.6 / 0.2 / 0.5 —— 同一模型、同一对手预算下的波动幅度是 ±0.3。`best_win_ratio` 在这种噪声下选出来的「最佳模型」基本是运气最好的那次。

而且**对手预算在变**（1000 → 5000 playout），胜率数字跨行不可比。`RESULTS.md` 里那句「比较胜率时需要同时看搜索预算」是对的，但既然如此，评估协议本身就该固定预算 + 增加局数（50–100 局）。

### 6. replay buffer 太小 —— `train.py:38`

`buffer_size = 10000`。退化后每局 18 手 × 9（1 原始 + 8 对称）= 162 条，buffer 只覆盖约 60 局。策略一旦绕进局部最优，数据里全是这套走法的自我强化，没有历史对比来把它拉出来。上游用 10000 是因为 6×6 每局数据量更小、且它只训 1500 局。

---

## 五、性能（时间基本都花在这里）

| 位置 | 问题 | 影响 |
| --- | --- | --- |
| `mcts_alphaZero.py:146` | 每次 playout 都 `copy.deepcopy(state)` | 400 次/步 × ~30 步 ≈ 12000 次深拷贝/局 |
| `policy_value_net_pytorch.py:106` | 推理没包 `torch.no_grad()` | 每次前向都建计算图，显存峰值高、变慢 |
| 同上 | 每次前向只喂 1 个样本 | GPU 利用率极低，batch=1 跑 2399 局 |
| `train.py:113` | 每个 epoch 都对**整个 batch** 再前向一次算 KL | 5 个 epoch → 多出 5 次 512 样本前向，纯为监控 |
| `game.py:92` | `has_a_winner` 每次重建 `set(range(W*H)) - set(availables)` | 纯 MCTS rollout 每步都调，O(N²)，评估的主要开销 |
| `mcts_alphaZero.py:204-206` | 非 selfplay 时 `update_with_move(-1)` 丢弃整棵子树 | 人机对局每步重建 400 次搜索，白扔一半算力 |
| `mcts_pure.py:148` | rollout 用 `np.random.rand` 生成全表再取 max | 等价均匀随机，但每步 O(N) 分配 |

`has_a_winner` 那条尤其值得注意：`policy_evaluate` 每 50 局跑 10 局对 1000–5000 playout 的纯 MCTS，每个 playout 含一次到终局的随机 rollout（每步一次 `game_end`）。**47 次评估累计的 rollout 步数远超 2399 局自对弈的前向次数量级**。想加速，先改这里，比动网络结构有效得多。

---

## 六、优先级清单

| 优先级 | 动作 | 位置 |
| --- | --- | --- |
| P0 | 自对弈随机换边 | `game.py:194` → `init_board(np.random.randint(2))` |
| P0 | 验证价值头是否活：打印 `v.std()` / `(v*z).mean()` / fc2 范数 | `train.py:105` 附近 |
| P0 | 棋盘退回 8×8 或 6×6，`n_playout` 提到 800–1600 | `train.py:25-27, 36` |
| P1 | 修通道 3 语义（按 `current_player` 判断） | `game.py:73` |
| P1 | 温度衰减：`temp = 1.0 if len(states) < 10 else 1e-2` | `game.py:198` |
| P1 | `torch.load(..., map_location=...)` + CUDA 可用性检查 | `policy_value_net_pytorch.py:77`, `train.py:55,60` |
| P1 | 评估局数 10 → 50，对手预算固定 | `train.py:150` |
| P2 | 修 `% self.height` → `% self.width`，或断言 `width == height` | `game.py:66-72` |
| P2 | 去掉/重标定 KL 自适应；`min(..., 10)` | `train.py:118-124` |
| P2 | 推理加 `torch.no_grad()`；`has_a_winner` 增量化 | `policy_value_net_pytorch.py:80,96`; `game.py:86` |
| P3 | `human_play.py` 支持 PyTorch 模型加载 | `human_play.py:52-72` |

---

## 七、`RESULTS.md` 的事实核对

- **「`current_policy.model` 和 `best_policy.model` 的模型权重相同」—— 对根目录那两个文件不成立。** 按 zip 成员逐字节比对：根目录两个模型 16 个权重张量**全部不同**；`gpu_run_20261113_104802/` 里那一对才是 16/16 完全相同。可能是保存时机记混了，或者拿错了目录。
- **「价值预测均约 0.0225」这条被当成了结果记录，实际是全流程最严重的问题**，应该在正文里点明是失效，而不是放在「结果」段落里。
- 训练在第 2399 局中断（`game_batch_num = 3000`），第 2350 局之后没有再保存模型。所以 `RESULTS.md` 描述的「第 2350 局的模型」确实是最后一个可用的 checkpoint。
- CPU 那轮（第 2504–3000 局）的 `lr_multiplier` 恒为 11.391、`explained_var` 恒为 0.000 —— 这一轮从头到尾没有产出有效模型，不建议作为对照实验引用，除非明确说明它是在错误的学习率下跑的。

---

## 八、修复与验证（2026-11-16）

### 已改动的代码

| 文件 | 改动 | 对应条目 | 是否影响结论 |
| --- | --- | --- | --- |
| `game.py:56-77` | 状态张量轴顺序改为 (通道, 高, 宽)；列坐标 `% self.width`；颜色通道按 `current_player` 判断 | 三.1 / 三.2 | 修正正确性，不是训练失败的成因 |
| `game.py:190` | `start_self_play` 增加 `start_player=None`，默认随机换边 | 四.1 | **消融显示非必需**，见下 |
| `policy_value_net_pytorch.py:77` | `torch.load` 加 `map_location` | 二.3 | 只影响跨机器加载 |
| `policy_value_net_pytorch.py:83,101` | 推理包 `no_grad`，去掉 `Variable` | 五 | 只影响速度 |
| `policy_value_net_pytorch.py:104`、`policy_value_net_numpy.py:101` | reshape 轴顺序改为 (H, W) | 三.1 | 修正正确性 |

方棋盘下轴顺序与原来完全等价，已训练的三个 checkpoint 加载后输出不变（空盘值 0.0225 / 0.0567 未变），这些修改不会让旧模型失效。

新增 `train_fixed.py`：可配棋盘、固定学习率、温度按步数衰减、π 恒取访问数比例、逐步记录 `v` 的均值/标准差/与 `z` 的相关性，`--fixed-first` 用于消融。

### 训练结果

| 配置 | 局数 | 每步模拟 | corr(v,z) | v_std | value_loss |
| --- | ---: | ---: | --- | ---: | ---: |
| 原版（10×10，GPU 日志） | 2399 | 400 | 0 → **0** | 0.0000 | 1.00 → 1.00 |
| 修复版（6×6 连四，换边） | 150 | 200 | 0.45 → **0.75** | 0.34 → 0.72 | 0.73 → 0.42 |
| 修复版（8×8 连五，换边） | 100 | 200 | 0.36 → **0.50** | 0.10 → 0.40 | 1.01 → 0.76 |
| 修复版（10×10，固定先手） | 60 | 400 | 0.89 → **0.81** | 0.89 → 0.75 | 0.23 → 0.39 |

独立验证（40 个随机局面）：`fixed_policy`（6×6）v_std 0.70、`ablate_policy`（6×6）0.72、`fixed8_policy`（8×8）0.38；原版三个 checkpoint 全是 0.0000。

### 消融：到底哪一处改动起作用

6×6 连四，150 局，200 模拟/步，唯一变量是「换边」和「温度衰减」：

| 换边 | 温度衰减 | corr(v,z) | v_std |
| :---: | :---: | ---: | ---: |
| ✓ | ✓ | 0.75 | 0.72 |
| ✗ | ✓ | **0.83** | 0.83 |
| ✗ | ✗ | 0.74 | 0.66 |

**换边与否、温度衰减与否，都能让价值头活过来。** 我上一轮把「随机换边」当成根因是错的 —— 那个判断没有对照实验支撑，被自己的消融否掉了。6×6 本身容错性太强，三组都能在 20 局内突破，无法区分变量。

判别性实验在 10×10 上：**同样的棋盘、同样的固定先手、同样 400 模拟/步**，只把学习率从原版的 1.76e-4 换成 2e-3，`corr(v,z)` 从 0 变成第 10 局就有 0.89、第 60 局 0.81。

### 根因：KL 自适应把学习率锁死在 0.088×

`train.py:118-124` 的逻辑：

```python
if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:      # kl > 0.04
    self.lr_multiplier /= 1.5
elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:     # kl < 0.01
    self.lr_multiplier *= 1.5
```

`kl_targ = 0.02`，于是「降」的阈值是 0.04，「升」的阈值是 0.01。GPU 日志里 KL 的全程均值是 0.025 —— **正好卡在两个阈值之间，两条分支都不触发，`lr_multiplier` 永久冻结**。

冻结值是 0.088，即有效学习率 `2e-3 × 0.088 = 1.76e-4`。它是训练早期几次 KL > 0.04 一路除下来的；`> 0.1` 的下界让它在 0.088 处停住，此后再也涨不回去。

在这个学习率上，`val_fc1/val_fc2` 在 2399 局里累积的更新量不足以让 `tanh` 脱离线性区，价值头就停在「输出 batch 均值」上 —— 也就是 `explained_var ≡ 0`、`value_loss ≡ 1.000`。

这条机制同时解释了 CPU 那轮的镜像故障：那轮 KL 只有 0.0004，远低于 0.01，「升」的分支被反复触发，`lr_multiplier` 从 1.0 涨到 11.391（下界判断写成 `< 10` 而非 `min(...)`，实际上限是 15），有效学习率变成 0.0228，比初始值大 11 倍。**同一段代码，一轮把 lr 压到 1/11，一轮放大到 11 倍，两轮都没训出来。**

### 对局对比

| | 原版 10×10 | 修复版 8×8 |
| --- | --- | --- |
| 手数 | 24（后手斜五） | 15（先手竖五） |
| 网络价值 v | 24 手恒为 +0.0225 | −0.08 → +0.82 |
| 首选点访问占比 | 0.02–0.12（前期） | 0.05–0.86 |

原版那局在 `demo_game.html`，修复版 8×8 在 `demo_game_fixed.html`，6×6 在 `demo_game_fixed_6x6.html`。

### 结论

价值头失效的根因不是棋盘太大、不是训练不够久、也不是没换边，而是 **`kl_targ` 与这套超参不匹配导致学习率被锁死在一个数量级以下**。修好这一点，10×10 上 10 局就能让 `corr(v,z)` 到 0.89。

`kl_targ = 0.02` 是上游按 6×6、batch 512、5 epoch 调的。换到 10×10 之后单次更新的 KL 天然落在 0.02–0.03，正好卡死这个控制器。要么把 `kl_targ` 提到与配置匹配（10×10 下应显著大于 0.03），要么直接固定学习率 —— 后者更省事，也更容易复现。

### 人机对战

`human_play.py` 已重写：按文件头自动识别模型格式、从权重推断棋盘尺寸、默认加载棋力最强的模型，AI 每步会打印网络评估和 top-3 候选。

```bash
python human_play.py                                    # 默认 best_policy_8_8_5，你来执 X
python human_play.py --model ablate10_policy.model      # 换 10×10 修复版（很弱）
python human_play.py --playout 800 --first ai           # 加长搜索，让 AI 先手
```

输入 `列,行`（都从 0 开始，左下角是 `0,0`），`q` 退出。

### 各模型对同棋盘纯 MCTS 的战绩（`eval_models.py`）

6 局，先手轮换，双方各 200 模拟/步：

| 模型 | 棋盘 | 战绩 | 得分 | 平均手数 | 用时 |
| --- | --- | --- | ---: | ---: | ---: |
| `best_policy_8_8_5.model` | 8×8 | 6 胜 0 负 | 1.00 | 9 | 20s |
| `best_policy_6_6_4.model` | 6×6 | 6 胜 0 负 | 1.00 | 8 | 9s |
| `best_policy.model`（自己 2399 局） | 10×10 | 4 胜 2 负 | 0.67 | 16 | 51s |
| `ablate10_policy.model`（修复版 60 局） | 10×10 | 0 胜 6 负 | 0.00 | 20 | 40s |

要下棋就用上游那两个：训练充分，9 手内结束战斗。自己训的 2399 局那份能赢纯 MCTS，但价值头失效导致它不会判断形势，遇到真会下的人撑不住。修复版 60 局这个连纯 MCTS 都打不过 —— 它证明了价值头能复活，但不代表棋力够用。

### 图形化对战（`web_play.py`）

终端版能用，但看盘面费眼、看不到 AI 在想什么。`web_play.py` 起一个本地 HTTP 服务，浏览器里下：

```bash
python web_play.py                                  # http://127.0.0.1:8777
python web_play.py --playout 800 --model best_policy.model
python web_play.py --port 9000 --no-browser
```

MCTS 留在 Python 进程（模型是 torch/numpy 的，浏览器跑不动），前端只做渲染。对局状态由服务端单例持有，所以刷新页面不会丢局。

界面除了棋盘，还暴露三样在终端里看不到的东西：

| 面板 | 内容 | 为什么值得看 |
| --- | --- | --- |
| AI 形势判断 | 根节点网络评估 `v`、模拟次数、耗时 | 价值头坏没坏，一眼就能看出来 |
| 候选分布 | 搜索真正访问过的点、访问数占比 | 网络先验有多尖/多平，直接可见 |
| 搜索覆盖 | 「进入搜索 N / 空白点」+ 前三占比 | 衡量 MCTS 到底有没有在比较 |

这三个指标把两份模型的差别摆到了台面上：

- `best_policy_8_8_5`（8×8，训练充分）：`v` 在 −0.91 ~ +0.99 之间大幅摆动；400 模拟下只有 **2~4 个点**被真正访问过，其余 50 多个合法点的访问数都是 0。网络先验极尖，MCTS 基本在确认网络的首选。
- `best_policy.model`（10×10，价值头失效）：`v` 恒定在 **+0.057 附近**，400 模拟下 **84 / 96 个点**都被访问过，前三合计只占 **6%**。搜索退化成近似均匀采样，因为价值头给不出区分度，Q 全挤在 0 附近。

也就是说，10×10 那份模型的 MCTS 实际上在做「把所有点都试一遍」，而不是「挑几个点深算」。这是 `explained_var` 恒为 0 在棋力上的直接后果，比看曲线更直观。

操作：点棋盘落子，`重新开始 / 撤销一手 / AI 提示`，`叠加` 按钮在「搜索分布 / AI 提示 / 关闭」之间切换。撤销会一直回退到轮到你为止（AI 先手时也一样）。

#### 在别的设备上打开（手机 / 平板 / 另一台电脑）

默认只监听 `127.0.0.1`，别的机器连不上。加 `--host 0.0.0.0` 开放到局域网：

```bash
python web_play.py --host 0.0.0.0
```

```
模型 best_policy_8_8_5.model | NumPy (Theano 权重) | 8x8 / 5 连
本机打开   http://127.0.0.1:8777/
局域网访问（手机/平板要和这台机器在同一个 WiFi 或热点下）:
   wlan0    http://10.4.178.31:8777/?t=MjktFtgE
```

把带 `?t=` 的链接发到手机上打开即可。这条命令有三个设计决定：

1. **局域网访问必须带令牌**，回环地址免检。不带令牌连页面都拿不到（`401`），避免同网段任何人扫到端口就能接管棋局。带 `?t=` 首次打开时服务端种一个 `gp_tok` cookie，之后页面里的 `fetch` 自动带上，不用每次拼参数。令牌每次启动重新生成，Ctrl-C 就失效。
2. **不自动选「出口 IP」**。用 UDP `connect` 探出口的习惯写法在这台机器上会返回 `198.18.0.1` —— Clash/TUN 类代理的假地址，手机根本连不上。改成读 `ip -4 addr`，按网卡名排序（`wl*` 优先），并排掉 `docker0`/`br-*`/`virbr*`/`tun*`/`198.18.*`。
3. **探测端口必须设 `SO_REUSEADDR`**。上一次运行留下的 keep-alive 连接会停在 `CLOSE-WAIT`/`FIN-WAIT-2`，不设这项就绑不上 `8777`，服务会被无声挤到 `8778`，而 `HTTPServer` 自己设了 `allow_reuse_address`，本来是可以绑的。

**注意：服务端只有一个 `Session`，所有设备看到的是同一局。** 手机上点「重新开始」会把电脑上那盘也清掉。这是有意的（可以一台设备落子、另一台看盘），不是 bug；要各下各的就得开两个进程占两个端口。

校园网常见客户端隔离（AP isolation），手机和电脑即使连同一个 SSID 也互 ping 不通。遇到这种情况让电脑连手机热点，再从手机访问电脑的新 IP；或者反过来用手机的局域网 IP。公网访问不在本脚本范围内 —— 那需要端口映射或反向代理，别为了下一盘棋把 `8777` 直接暴露到公网。


