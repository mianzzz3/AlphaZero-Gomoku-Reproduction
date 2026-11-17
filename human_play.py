# -*- coding: utf-8 -*-
"""人机对战。

输入格式：列,行（例如 4,4 表示第 4 列第 4 行，行列都从 0 开始，棋盘左下角是 0,0）。
输入 q 退出。

用法：
    python human_play.py                          # 自动挑最新模型，你先手
    python human_play.py --model ablate10_policy.model
    python human_play.py --playout 800 --first ai
"""

from __future__ import print_function

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import Board                     # noqa: E402
from mcts_alphaZero import MCTSPlayer            # noqa: E402
from demo_game import load_backend, make_policy_fn  # noqa: E402

# 按已知棋力排序：上游训练好的模型最强，自己训的按局数。
MODEL_PREFERENCE = (
    'best_policy_8_8_5.model',   # 上游 8x8 连五，训练充分，棋力最可靠
    'best_policy.model',         # 自己 10x10 / 2399 局（价值头失效，靠策略先验）
    'best_policy_6_6_4.model',   # 上游 6x6 连四
    'ablate10_policy.model',     # 修复版 10x10，仅 60 局
    'fixed8_policy.model',       # 修复版 8x8，100 局
    'fixed_policy.model',        # 修复版 6x6，150 局
)


class Tracer(object):
    """包一层策略函数，记录每步根节点（第一次调用）的评估结果。"""

    def __init__(self, fn, side):
        self.fn = fn
        self.side = side
        self.root = None

    def reset(self):
        self.root = None

    def __call__(self, board):
        probs, value = self.fn(board)
        probs = list(probs)
        if self.root is None:
            self.root = (probs, float(value))
        return probs, value

    def summary(self):
        if not self.root:
            return ''
        probs, value = self.root
        top = sorted(probs, key=lambda p: -p[1])[:3]
        cand = '  '.join('(%d,%d) %.2f' % (a % self.side, a // self.side, p)
                         for a, p in top)
        return '  网络评估 v = %+.3f   候选: %s' % (value, cand)


class Human(object):
    """人类玩家。"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        while True:
            try:
                raw = input('Your move: ').strip()
            except EOFError:
                raise SystemExit('\n已退出')
            if raw in ('q', 'quit', 'exit'):
                raise SystemExit('已退出')
            try:
                loc = [int(n) for n in raw.replace('，', ',').split(',')]
            except ValueError:
                print('  格式是 列,行，例如 4,4')
                continue
            move = board.location_to_move(loc)
            if move == -1 or move not in board.availables:
                print('  无效落点，或该位置已有子')
                continue
            return move

    def __str__(self):
        return 'Human {}'.format(self.player)


def normalise(text):
    return ' '.join(text.split())


def print_board(board, last_move=-1):
    """紧凑棋盘：一行一格，比 Game.graphic 省一半屏幕。"""
    side = board.width
    rows = ['    ' + ' '.join('%2d' % j for j in range(side))]
    for i in range(side - 1, -1, -1):
        cells = []
        for j in range(side):
            mv = i * side + j
            p = board.states.get(mv, 0)
            ch = '.' if p == 0 else ('X' if p == 1 else 'O')
            cells.append('%2s' % (ch.lower() if mv == last_move else ch))
        rows.append('%3d ' % i + ''.join(cells))
    print('\n'.join(rows))


def run(model_path, n_playout, human_first, c_puct):
    net, side, n_in_row, backend = load_backend(model_path)
    board = Board(width=side, height=side, n_in_row=n_in_row)

    tracer = Tracer(make_policy_fn(net), side)
    ai = MCTSPlayer(tracer, c_puct=c_puct, n_playout=n_playout)
    human = Human()

    board.init_board(0)          # player1 恒为先手，谁执子由 --first 决定
    hp, ap = (1, 2) if human_first else (2, 1)
    human.set_player_ind(hp)
    ai.set_player_ind(ap)
    mark = {1: 'X', 2: 'O'}

    print('=' * 46)
    print('模型  :', os.path.basename(model_path), '|', backend)
    print('棋盘  : %dx%d，%d 连获胜' % (side, side, n_in_row))
    print('每步  : %d 次 MCTS 模拟' % n_playout)
    print('你执  : %s（player%d）' % (mark[hp], hp))
    print('输入  : 列,行  例如 4,4；q 退出')
    print('=' * 46)

    ply = 0
    last = -1
    while True:
        print()
        print_board(board, last)
        cur = board.current_player
        if cur == hp:
            print('\n第 %d 手 · 你（%s）' % (ply + 1, mark[hp]))
            move = human.get_action(board)
        else:
            print('\n第 %d 手 · AI（%s）思考中 ...' % (ply + 1, mark[ap]))
            tracer.reset()
            t0 = time.time()
            move = ai.get_action(board)
            print('  AI 落子 (%d,%d)   用时 %.1fs' % (
                move % side, move // side, time.time() - t0))
            print(normalise(tracer.summary()))

        board.do_move(move)
        last = move
        ply += 1

        end, winner = board.game_end()
        if end:
            print()
            print_board(board, last)
            if winner == -1:
                print('平局')
            elif winner == hp:
                print('你赢了')
            else:
                print('AI 赢了')
            return winner


def pick_model():
    for name in MODEL_PREFERENCE:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            return path
    return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--playout', type=int, default=400)
    ap.add_argument('--c-puct', type=float, default=5.0)
    ap.add_argument('--first', choices=('human', 'ai'), default='human')
    a = ap.parse_args()

    model = a.model or pick_model()
    if not model:
        raise SystemExit('找不到可用的模型文件')
    if not os.path.isabs(model):
        model = os.path.join(HERE, model)
    if not os.path.exists(model):
        raise SystemExit('模型不存在: %s' % model)

    try:
        run(model, a.playout, a.first == 'human', a.c_puct)
    except KeyboardInterrupt:
        print('\n已退出')
