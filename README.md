# AlphaZero 五子棋复现

在 [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) 的基础上做三件事：

1. **复现 10×10 连五的自对弈训练**（GPU，2399 局），并定位到价值网络失效的真正根因；
2. **修掉 `game.py` 里三处会静默算错的坐标/通道问题**，以及几处直接抛异常的加载路径；
3. **补上三套工具**：图形化人机对战、单局决策回放、训练过程复盘。

代码审查全文见 [CODE_REVIEW.md](CODE_REVIEW.md)（含量化证据与消融实验），训练实验记录见 [RESULTS.md](RESULTS.md)。

---

## 快速开始

| 我想…… | 命令 |
| --- | --- |
| 在浏览器里和 AI 下棋 | `python web_play.py`，然后打开 <http://127.0.0.1:8777> |
| 让手机/平板也来一盘 | `python web_play.py --host 0.0.0.0`，用它打印的局域网链接 |
| 在终端里下棋 | `python human_play.py` |
| 看 AI 每一步在想什么 | `python demo_game.py && python render_game.py` → 打开 `demo_game.html` |
| 复盘那次失败训练 | 直接打开 `training_report.html`（单文件，无外部依赖） |
| 评测某份模型的棋力 | `python eval_models.py --games 6` |
| 自己训一版 | `python train_fixed.py --width 6 --n-in-row 4 --games 150` |

推理和图形界面只需要 **NumPy**；只有加载 `.model`（PyTorch zip 格式）或训练时才需要 **PyTorch**。Python 3.8+（实测 3.13）。

---

## 哪份模型能打

6 局，先手轮换，对手是同棋盘的纯 MCTS（每步 200 次模拟）：

| 文件 | 棋盘 | 来源 | 战绩 | 得分 |
| --- | --- | --- | :---: | ---: |
| `best_policy_8_8_5.model` | 8×8 连五 | 上游 Theano/Lasagne，pickle | 6 胜 0 负 | 1.00 |
| `best_policy_6_6_4.model` | 6×6 连四 | 上游 Theano/Lasagne，pickle | 6 胜 0 负 | 1.00 |
| `best_policy.model` | 10×10 连五 | **本仓库**训 2399 局，PyTorch | 4 胜 2 负 | 0.67 |
| `ablate10_policy.model` | 10×10 连五 | 修复版只训 60 局 | 0 胜 6 负 | 0.00 |

**想下棋就用前两个。** 自己训的那份策略头是好的（能赢纯 MCTS），但价值头失效（见下节），不会判断形势，遇到真人撑不住。`fixed*_policy.model` 系列是修复版训练的输出，只跑了 60–150 局，仅用于验证价值头能复活，不是可用的棋力。

`*.model2` 是同名模型的另一份存档（同为 Theano pickle 格式），未做交叉验证，不建议优先使用。

---

## 训练为什么失败

2399 局之后的指标是这样的：

- `value_loss` 恒等于 **1.000**（常数预测的 MSE 下界）；
- `explained_var` 只在最初 24 个 batch 有意义，此后 |值| ≤ 0.007，**98.7% 精确等于 0**；
- 对 256 个随机合法局面的价值预测全部约 **0.0225**（`RESULTS.md` 记的就是这个现象）。

也就是说，后 2370 局都在训练一个已经死了的价值头。但策略头是活的：`policy_loss` 从 4.60 降到 2.29、`entropy` 从 3.85 降到 2.31。

**根因不是棋盘太大，也不是训练不够久，是 `train.py` 里的 KL 自适应把学习率锁死了。** `kl_targ = 0.02` 对应降 lr 阈值 0.04、升 lr 阈值 0.01，而实测单次更新的 KL 稳定在 0.025 —— 两个分支都不触发，`lr_multiplier` 冻结在 0.088，有效学习率只有 `1.76e-4`。这个 `kl_targ` 是上游按 6×6 / batch 512 / 5 epoch 调的，换到 10×10 就恰好卡死。

`train_fixed.py` 的验证结果：换成固定学习率 `2e-3` 后，10×10 上第 10 局 `corr(v,z)` 就到 0.89、第 60 局 0.81，6×6 上 150 局到 0.75。价值头能救回来 —— 但还没跑到能打的局数。

`train.py` 本身保留原样，以便复现这次失败。

---

## 图形界面长什么样

`web_play.py` 起一个本地 HTTP 服务，MCTS 留在 Python 进程，浏览器只负责渲染。除了棋盘，它还暴露三样终端里看不到的东西：

| 面板 | 内容 |
| --- | --- |
| AI 形势判断 | 根节点网络评估 `v`、模拟次数、每步耗时 |
| 候选分布 | 搜索真正访问过的点、访问数占比 |
| 搜索覆盖 | 「进入搜索 N / 空白点」+ 前三合计 |

这三个指标把两份模型的差别摆到了台面上：8×8 上游模型 400 次模拟只访问 **2~4 个点**（先验极尖，MCTS 基本在确认网络首选）；10×10 那份（价值头失效）访问 **84/96 个点**、前三合计仅 6%、`v` 恒定在 +0.057 —— 它的 MCTS 实际在做均匀采样，而不是挑点深算。

用法细节（令牌、局域网、叠加层）见 [CODE_REVIEW.md 第九节](CODE_REVIEW.md)。

---

## 目录

| 文件 | 作用 |
| --- | --- |
| `game.py` | 棋盘状态、胜负判定、`current_state()` 张量构造 |
| `mcts_alphaZero.py` | PUCT 搜索与 `MCTSPlayer` |
| `mcts_pure.py` | 纯蒙特卡洛对手（评测基准） |
| `policy_value_net_pytorch.py` | 网络定义与训练步（本仓库实际使用） |
| `policy_value_net_numpy.py` | 用 NumPy 手写前向，加载上游 Theano 权重 |
| `policy_value_net_{theano,tensorflow,keras}.py` | 上游其他框架实现，未使用 |
| `train.py` | 原版训练流程（保留失败配置） |
| `train_fixed.py` | 修复版：随机换边、温度衰减、固定学习率、逐步记录 `v` 相关性 |
| `web_play.py` | 图形化人机对战（本地 HTTP + 内嵌 Canvas 前端） |
| `human_play.py` | 终端人机对战，自动识别模型格式与棋盘尺寸 |
| `demo_game.py` | 跑一局自对弈，导出每步的先验/搜索/`v` 到 JSON |
| `render_game.py` | 把上述 JSON 渲染成可逐步回放的 HTML |
| `make_report.py` | 解析训练日志与 CSV，生成 `training_report.html` |
| `eval_models.py` | 让模型与同棋盘纯 MCTS 对打，算得分 |

---

## 依赖

```bash
pip install numpy            # 推理 / 图形界面 / 终端对战
pip install torch            # 加载 .model（PyTorch zip）与训练
```

Theano / Lasagne / TensorFlow 的实现保留在仓库里但没跑通，需要的话自行补环境。

---

## 训练

```bash
# 修复版：6×6 连四，150 局，单核 CPU 几分钟
python train_fixed.py --width 6 --n-in-row 4 --games 150 --out-prefix fixed

# 10×10 连五
python train_fixed.py --width 10 --n-in-row 5 --games 500 --playout 400 --out-prefix fixed10
```

每局结束会打印 `v_std` / `v_corr` / `vloss` / `kl` —— **`v_corr` 是判断价值头死活的唯一可靠指标**，看 `value_loss` 会被常数预测骗过去（它恒等于 1.000 时看起来还挺像回事）。`v_std` 掉到 1e-4 量级、`v_corr` 贴 0，就是价值头死了。

原版训练流程用 `python train.py`，配置在文件顶部的 `TrainPipeline.__init__` 里（默认 10×10、400 playout、3000 局）。

---

## 已知问题

- `RESULTS.md` 说「`current_policy.model` 与 `best_policy.model` 权重相同」—— 这只在 `gpu_run_20261113_104802/` 那一对成立；根目录那一对的 16 个参数成员逐项 md5 全不相同。文档未修订。
- `train.py` 的 KL 自适应缺陷保持原样，没有修。
- 修复版最长只跑到 150 局（10×10 到 60 局），没有产出真正能打的 10×10 模型。
- `*.model2` 与同名 `.model` 的权重关系未验证。
- 网络是纯 MLP（3 层卷积 + 全连接），不是论文里的残差结构；8×8 以上棋力上限受此限制。

---

## 引用与许可

上游项目 [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)，MIT License，Copyright (c) 2017 junxiaosong。本仓库的修改同样以 MIT 发布。

1. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero)
2. *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero)
