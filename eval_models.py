# -*- coding: utf-8 -*-
"""让候选模型跟同棋盘的纯 MCTS 对打，用胜率决定该默认加载谁。

用法：
    python eval_models.py --games 2 --playout 200
"""

from __future__ import print_function

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import Board, Game                        # noqa: E402
from mcts_alphaZero import MCTSPlayer               # noqa: E402
from mcts_pure import MCTSPlayer as PureMCTS        # noqa: E402
from demo_game import load_backend, make_policy_fn  # noqa: E402

CANDIDATES = [
    'best_policy_8_8_5.model',
    'best_policy.model',
    'ablate10_policy.model',
    'best_policy_6_6_4.model',
]


def evaluate(model_path, n_games, n_playout, pure_playout):
    net, side, n_in_row, backend = load_backend(model_path)
    board = Board(width=side, height=side, n_in_row=n_in_row)
    game = Game(board)
    policy_fn = make_policy_fn(net)

    wins = losses = ties = 0
    plies = []
    t0 = time.time()
    for i in range(n_games):
        ai = MCTSPlayer(policy_fn, c_puct=5, n_playout=n_playout)
        pure = PureMCTS(c_puct=5, n_playout=pure_playout)
        winner = game.start_play(ai, pure, start_player=i % 2, is_shown=0)
        plies.append(len(board.states))
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            ties += 1
    score = (wins + 0.5 * ties) / max(1, n_games)
    return dict(model=os.path.basename(model_path), side=side, n=n_in_row,
                backend=backend, wins=wins, losses=losses, ties=ties,
                score=score, avg_plies=sum(plies) / max(1, len(plies)),
                sec=time.time() - t0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=2)
    ap.add_argument('--playout', type=int, default=200)
    ap.add_argument('--pure-playout', type=int, default=None)
    a = ap.parse_args()
    pure_pl = a.pure_playout or a.playout

    print('每模型 %d 局，AI %d 模拟/步，纯 MCTS %d 模拟/步\n'
          % (a.games, a.playout, pure_pl))
    print('%-26s %-6s %-22s %s' % ('模型', '棋盘', '战绩', '用时'))
    results = []
    for name in CANDIDATES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        r = evaluate(path, a.games, a.playout, pure_pl)
        results.append(r)
        print('%-26s %-6s %d 胜 %d 负 %d 平  得分 %.2f  平均 %d 手  %.0fs'
              % (r['model'], '%dx%d' % (r['side'], r['side']),
                 r['wins'], r['losses'], r['ties'], r['score'],
                 r['avg_plies'], r['sec']), flush=True)

    if results:
        best = max(results, key=lambda r: r['score'])
        print('\n最强: %s（得分 %.2f）' % (best['model'], best['score']))
