# -*- coding: utf-8 -*-
"""用修好的配置重训一次，专门验证价值头能否学起来。

和 train.py 的差别（全部对应 CODE_REVIEW.md 里的条目）：
  1. 自对弈随机换边            —— 修 game.py:194 的分布单一
  2. 温度按步数衰减            —— 前 N 步 temp=1，之后趋近 0
  3. π 始终用访问数比例        —— 训练目标不再随温度退化
  4. 固定学习率，不做 KL 自适应 —— 原逻辑在 batch 512 / 5 epoch 下两头跑偏
  5. 每步记录 v 的均值/标准差/与 z 的相关 —— 价值头死没死一眼可见
  6. 推理走 no_grad、棋盘尺寸可配

用法：
    python train_fixed.py --width 6 --n-in-row 4 --games 120 --playout 200
"""

from __future__ import print_function

import argparse
import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import Board, Game                      # noqa: E402
from mcts_alphaZero import MCTS                   # noqa: E402
from policy_value_net_pytorch import PolicyValueNet  # noqa: E402


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()


class FixedPipeline(object):
    def __init__(self, width, n_in_row, n_playout, games, lr,
                 buffer_size=20000, batch_size=512, epochs=5,
                 temp_moves=10, temp_late=1e-2, out_prefix='fixed',
                 swap_side=True):
        self.swap_side = swap_side
        self.width = width
        self.n_in_row = n_in_row
        self.n_playout = n_playout
        self.games = games
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.temp_moves = temp_moves
        self.temp_late = temp_late
        self.c_puct = 5.0
        self.out_prefix = out_prefix

        self.board = Board(width=width, height=width, n_in_row=n_in_row)
        self.game = Game(self.board)
        self.net = PolicyValueNet(width, width, use_gpu=False)
        self.mcts = MCTS(self.net.policy_value_fn,
                         c_puct=self.c_puct, n_playout=n_playout)
        self.data_buffer = deque(maxlen=buffer_size)

        self.rows = []
        self.csv_path = os.path.join(HERE, '%s_metrics.csv' % out_prefix)
        with open(self.csv_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['game', 'ep_len', 'first_player', 'winner',
                        'loss', 'policy_loss', 'value_loss', 'entropy', 'kl',
                        'v_mean', 'v_std', 'v_corr', 'lr', 'elapsed'])

    # ---------- 自对弈 ----------
    def selfplay(self):
        board = self.board
        start_player = int(np.random.randint(2)) if self.swap_side else 0
        board.init_board(start_player)
        states, pis, current_players = [], [], []
        t0 = time.time()

        while True:
            step = len(states)
            temp = 1.0 if step < self.temp_moves else self.temp_late

            # π 永远用访问数比例（temp=1），只把降温用在采样上
            acts, pi = self.mcts.get_move_probs(board, temp=1.0)
            acts = list(acts)
            pi = np.asarray(pi, dtype=np.float64)

            states.append(board.current_state())
            pis.append(np.asarray(self._to_full_pi(acts, pi)))
            current_players.append(board.current_player)

            if step < self.temp_moves:
                p_sample = pi
            else:
                p_sample = softmax(np.log(pi + 1e-10) / temp)

            noise = np.random.dirichlet(0.3 * np.ones(len(p_sample)))
            p_mix = 0.75 * p_sample + 0.25 * noise
            p_mix = np.clip(p_mix, 1e-10, None)
            p_mix /= p_mix.sum()

            k = int(np.random.choice(len(acts), p=p_mix))
            move = int(acts[k])

            self.mcts.update_with_move(move)
            board.do_move(move)

            end, winner = board.game_end()
            if end:
                z = np.zeros(len(current_players))
                if winner != -1:
                    z[np.array(current_players) == winner] = 1.0
                    z[np.array(current_players) != winner] = -1.0
                self.mcts.update_with_move(-1)
                return (states, pis, z, winner, start_player,
                        len(states), time.time() - t0)

    def _to_full_pi(self, acts, pi):
        full = np.zeros(self.width * self.width)
        full[acts] = pi
        return full

    # ---------- 对称增强 ----------
    def augment(self, states, pis, z):
        out = []
        for s, p, zi in zip(states, pis, z):
            for k in (1, 2, 3, 4):
                rs = np.array([np.rot90(x, k) for x in s])
                rp = np.rot90(p.reshape(self.width, self.width), k).flatten()
                out.append((rs, rp, zi))
                fs = np.array([np.fliplr(x) for x in rs])
                fp = np.fliplr(rp.reshape(self.width, self.width)).flatten()
                out.append((fs, fp, zi))
        return out

    # ---------- 更新 ----------
    def update(self):
        mini = random.sample(self.data_buffer, self.batch_size)
        sb = [d[0] for d in mini]
        pb = [d[1] for d in mini]
        zb = [d[2] for d in mini]

        z_arr = np.asarray(zb, dtype=np.float64)
        old_probs, old_v = self.net.policy_value(sb)

        loss = entropy = ploss = vloss = float('nan')
        for _ in range(self.epochs):
            loss, entropy, ploss, vloss = self.net.train_step(
                sb, pb, zb, self.lr, return_loss_components=True)

        new_probs, new_v = self.net.policy_value(sb)
        v = np.asarray(new_v, dtype=np.float64).ravel()
        kl = float(np.mean(np.sum(
            old_probs * (np.log(old_probs + 1e-10) -
                         np.log(new_probs + 1e-10)), axis=1)))
        v_std = float(v.std())
        if z_arr.std() > 1e-8 and v.std() > 1e-8:
            v_corr = float(np.corrcoef(v, z_arr)[0, 1])
        else:
            v_corr = 0.0
        return dict(loss=loss, policy_loss=ploss, value_loss=vloss,
                    entropy=entropy, kl=kl, v_mean=float(v.mean()),
                    v_std=v_std, v_corr=v_corr)

    # ---------- 主循环 ----------
    def run(self):
        t_all = time.time()
        for g in range(1, self.games + 1):
            states, pis, z, winner, first, ep_len, ep_sec = self.selfplay()
            data = self.augment(states, pis, z)
            self.data_buffer.extend(data)

            row = dict(game=g, ep_len=ep_len, first_player=first,
                       winner=winner, elapsed=round(ep_sec, 2))
            if len(self.data_buffer) > self.batch_size:
                row.update(self.update())
                row['lr'] = self.lr

            with open(self.csv_path, 'a', newline='') as fh:
                csv.writer(fh).writerow([
                    row.get('game'), row.get('ep_len'),
                    row.get('first_player'), row.get('winner'),
                    row.get('loss'), row.get('policy_loss'),
                    row.get('value_loss'), row.get('entropy'),
                    row.get('kl'), row.get('v_mean'), row.get('v_std'),
                    row.get('v_corr'), row.get('lr'), row.get('elapsed')])

            if g % 10 == 0 or g == 1:
                if 'v_corr' not in row:
                    # buffer 还没攒够一个 batch，这一局只自对弈不训练。
                    # 直接打 nan 会让人以为训练崩了。
                    print('g:%3d ep:%2d first:%d win:%2d  样本 %d/%d，'
                          '还没攒够一个 batch，跳过训练'
                          % (g, ep_len, first, winner,
                             len(self.data_buffer), self.batch_size),
                          flush=True)
                else:
                    print('g:%3d ep:%2d first:%d win:%2d v_std:%.4f '
                          'v_corr:%+.3f vloss:%.3f ploss:%.3f ent:%.2f kl:%.4f'
                          % (g, ep_len, first, winner,
                             row['v_std'], row['v_corr'], row['value_loss'],
                             row['policy_loss'], row['entropy'], row['kl']),
                          flush=True)

        self.net.save_model(os.path.join(HERE, '%s_policy.model' % self.out_prefix))
        print('total %.1fs -> %s_policy.model' % (time.time() - t_all,
                                                  self.out_prefix))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=6)
    ap.add_argument('--n-in-row', type=int, default=4)
    ap.add_argument('--playout', type=int, default=200)
    ap.add_argument('--games', type=int, default=120)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--temp-moves', type=int, default=10)
    ap.add_argument('--out-prefix', default='fixed')
    ap.add_argument('--fixed-first', action='store_true',
                    help='固定 player1 先手，复现原版的数据分布（消融用）')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)

    FixedPipeline(a.width, a.n_in_row, a.playout, a.games, a.lr,
                  epochs=a.epochs, temp_moves=a.temp_moves,
                  out_prefix=a.out_prefix,
                  swap_side=not a.fixed_first).run()
