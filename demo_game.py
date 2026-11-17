# -*- coding: utf-8 -*-
"""让训练好的模型自对弈一局，把每一步的决策依据导出为 JSON。

用法：
    python demo_game.py                          # 自动挑选可用模型
    python demo_game.py --model best_policy.model --playout 400
    python demo_game.py --model best_policy_8_8_5.model

输出：demo_game.json（供 render_game.py 渲染）
"""

from __future__ import print_function

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import Board                       # noqa: E402
from mcts_alphaZero import MCTS              # noqa: E402


DEFAULT_N_IN_ROW = {6: 4, 8: 5, 10: 5}


def load_backend(model_path, force_side=None, force_n_in_row=None):
    """返回 (policy_value_fn, 几何信息, 后端描述)。优先 PyTorch，退回 numpy。"""
    name = os.path.basename(model_path)
    with open(model_path, 'rb') as fh:
        magic = fh.read(4)

    torch_model = magic == b'PK\x03\x04'
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    if torch_model and not has_torch:
        raise SystemExit(
            '这个模型是 PyTorch 格式（zip），但当前解释器没有 torch。\n'
            '改用 numpy 兼容的上游模型，例如 best_policy_8_8_5.model；\n'
            '或在装了 torch 的环境里运行。')

    if torch_model:
        from policy_value_net_pytorch import PolicyValueNet
        # batch=1 的前向在多线程下开销远大于计算本身
        torch.set_num_threads(1)
        state = torch.load(model_path, map_location='cpu')
        n_cells = int(state['act_fc1.weight'].shape[0])
        side = int(round(n_cells ** 0.5))
        if side * side != n_cells:
            raise SystemExit('无法从权重推断棋盘尺寸: %d' % n_cells)
        net = PolicyValueNet(side, side, use_gpu=False)
        # 不能把 model_file 直接交给 PolicyValueNet：它内部的 torch.load
        # 没有 map_location，加载 GPU 上保存的 checkpoint 会直接抛异常。
        net.policy_value_net.load_state_dict(state)
        m = re.search(r'_(\d+)_(\d+)_(\d+)\.model', name)
        n_in_row = (int(m.group(3)) if m
                    else (force_n_in_row or DEFAULT_N_IN_ROW.get(side, 5)))
        backend = 'PyTorch %s (CPU)' % torch.__version__
    else:
        import pickle
        from policy_value_net_numpy import PolicyValueNetNumpy
        try:
            params = pickle.load(open(model_path, 'rb'))
        except UnicodeDecodeError:
            params = pickle.load(open(model_path, 'rb'), encoding='bytes')
        # 上游命名 _W_H_N 直接给出几何；否则要求显式指定
        m = re.search(r'_(\d+)_(\d+)_(\d+)\.model', name)
        if m:
            side, n_in_row = int(m.group(1)), int(m.group(3))
        elif force_side:
            side, n_in_row = force_side, (force_n_in_row or
                                          DEFAULT_N_IN_ROW.get(force_side, 5))
        else:
            raise SystemExit(
                '无法从文件名推断棋盘尺寸，请用 --width / --n-in-row 指定')
        net = PolicyValueNetNumpy(side, side, params)
        backend = 'NumPy (Theano 权重)'

    return net, side, n_in_row, backend


def make_policy_fn(net):
    """包一层 no_grad：原始实现每次前向都在建计算图，MCTS 里纯浪费时间。"""
    raw = net.policy_value_fn
    if getattr(net, 'use_gpu', None) is None:
        return raw
    import torch

    def fn(board):
        with torch.no_grad():
            return raw(board)
    return fn


def top_k(pairs, k=5):
    ordered = sorted(pairs, key=lambda p: -p[1])[:k]
    return [[int(a), round(float(p), 4)] for a, p in ordered]


def play(model_path, n_playout, c_puct, seed, max_moves,
         force_side=None, force_n_in_row=None, out_name='demo_game.json'):
    net, side, n_in_row, backend = load_backend(model_path, force_side,
                                                force_n_in_row)
    policy_fn = make_policy_fn(net)

    board = Board(width=side, height=side, n_in_row=n_in_row)
    board.init_board(0)

    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    moves, t_start = [], time.time()

    while True:
        t0 = time.time()
        prior_pairs, net_value = policy_fn(board)
        prior_pairs = list(prior_pairs)

        mcts = MCTS(policy_fn, c_puct=c_puct, n_playout=n_playout)
        acts, probs = mcts.get_move_probs(board, temp=1.0)
        search_pairs = list(zip(acts, probs))

        order = np.argsort(np.asarray(probs))[::-1]
        best_i = int(order[0])
        move = int(acts[best_i])

        root_visits = sum(int(n._n_visits)
                          for n in mcts._root._children.values())
        root_q = None
        if move in mcts._root._children:
            root_q = float(mcts._root._children[move]._Q)

        h, w = divmod(move, side)
        moves.append({
            'i': len(moves),
            'player': int(board.current_player),
            'move': move,
            'h': int(h),
            'w': int(w),
            'net_value': round(float(net_value), 4),
            'root_q': None if root_q is None else round(root_q, 4),
            'root_visits': root_visits,
            'prior': top_k(prior_pairs, 5),
            'search': top_k(search_pairs, 5),
            'confidence': round(float(probs[best_i]), 4),
            'elapsed': round(time.time() - t0, 3),
        })

        board.do_move(move)
        end, winner = board.game_end()
        if end or len(moves) >= max_moves:
            if end:
                break
            winner = -1
            break

    final = []
    for h in range(side - 1, -1, -1):
        row = []
        for w in range(side):
            row.append(int(board.states.get(h * side + w, 0)))
        final.append(row)

    payload = {
        'meta': {
            'model': os.path.basename(model_path),
            'backend': backend,
            'board_width': side,
            'board_height': side,
            'n_in_row': n_in_row,
            'n_playout': n_playout,
            'c_puct': c_puct,
            'seed': seed,
            'winner': int(winner),
            'total_moves': len(moves),
            'elapsed': round(time.time() - t_start, 2),
        },
        'moves': moves,
        'final_board': final,
    }

    out = os.path.join(HERE, out_name)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    print('model   :', payload['meta']['model'], '|', backend)
    print('board   : %dx%d, %d in a row' % (side, side, n_in_row))
    print('playout : %d/step, %d plies, %.1fs' % (
        n_playout, len(moves), payload['meta']['elapsed']))
    print('winner  :', 'tie' if winner == -1 else 'player %d' % winner)
    print('written :', out)
    return payload


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--playout', type=int, default=400)
    ap.add_argument('--c-puct', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--max-moves', type=int, default=60)
    ap.add_argument('--width', type=int, default=None,
                    help='无法从文件名推断时的棋盘边长')
    ap.add_argument('--n-in-row', type=int, default=None)
    ap.add_argument('--out', default='demo_game.json')
    args = ap.parse_args()

    model = args.model
    if model is None:
        for cand in ('best_policy.model', 'best_policy_8_8_5.model',
                     'best_policy_6_6_4.model'):
            if os.path.exists(os.path.join(HERE, cand)):
                model = os.path.join(HERE, cand)
                break
    if not model or not os.path.exists(model):
        raise SystemExit('找不到模型文件')

    play(model, args.playout, args.c_puct, args.seed, args.max_moves,
         args.width, args.n_in_row, args.out)
