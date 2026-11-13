# AlphaZero 五子棋复现实验

## GPU 训练

- 棋盘为 10×10，连五获胜。策略价值网络使用 PyTorch，从随机权重开始训练，每步 MCTS 搜索 400 次。
- 计划训练 3000 局。终端关闭前完成 2399 局自对弈，日志记录了 2398 次网络更新和 47 次评估。
- 最后一次保存发生在第 2350 局。`current_policy.model` 和 `best_policy.model` 的模型权重相同。第 2351–2399 局有训练指标，未生成后续模型。

| 自对弈进度 | 纯 MCTS 对手每步搜索次数 | 评估胜负 |
| ---: | ---: | --- |
| 50 | 1000 | 0 胜 10 负 |
| 850 | 1000 | 10 胜 0 负 |
| 2300 | 4000 | 10 胜 0 负 |
| 2350 | 5000 | 9 胜 1 负 |

每次评估只下 10 局，先后手轮换。对手的搜索预算会随训练进度增加，因此比较胜率时需要同时看搜索预算。第 2301–2399 局的平均 `policy_loss` 为 2.246，平均 `value_loss` 为 0.999。第 2350 局模型对 256 个随机合法局面的价值预测均约为 0.0225。

- [完整 GPU 训练日志](gpu_run_20261113_104802/train_output.log)
- [第 2350 局模型](gpu_run_20261113_104802/current_policy.model) · [最佳模型](gpu_run_20261113_104802/best_policy.model)

## 早期 CPU 训练

- [终端日志](train_output.log)保留了第 2504–3000 局的输出。
- [训练指标](train_output_metrics.csv) · [评估结果](train_output_evaluations.csv) · [最后模型](current_policy.model) · [最佳模型](best_policy.model)

## 启动新的 GPU 训练

在装有 NumPy 和支持 CUDA 的 PyTorch 的环境中，从仓库根目录运行以下 Bash 命令。新目录用于保存本次训练日志和模型。

```bash
run_dir="gpu_run_$(date +%Y%m%d_%H%M%S)"
python -c 'import torch; assert torch.cuda.is_available()' && mkdir "$run_dir" && (
    cd "$run_dir" || exit 1
    set -o pipefail
    python -u ../train.py 2>&1 | tee train_output.log
)
```
