import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline


# Environment
class CoinToss(object):

    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("stpe countがMaxを超えてる！resetすべし！")

        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("アクション(コイン) {} は存在しない！".format(action))

        else:
            # action -> どのコインを選んだか
            head_prob = self.head_probs[action]
            # random.random() -> 0-1の少数を出力
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done

# Agent
class EpsilonGreedyAgent(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self, env):
        # Initialize estimation(価値見積)
        N = [0] * len(env)
        # ここでいうenv -> CoinToss
        self.V = [0] * len(env)
        env.reset()
        done = False
        rewards = []
        while not done:
            # 行動選択→step踏んでrewardとdoneもらう→rewardをrewardsに追加→学習
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average
        return rewards

# epsilonがいくつで、何回施行回数があった場合平均で何点とれるかをプロットする
def main():
    #probs = [random.random()]*4
    #env = CoinToss(probs)
    env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
    #env = CoinToss([0.0, 1.0])
    #env = CoinToss([1.0, 0.0])
    epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
    game_steps = list(range(10, 310, 10))
    # game_steps
    result = {}
    for e in epsilons:
        agent = EpsilonGreedyAgent(epsilon=e)
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            # 行動選択→step踏んでrewardとdoneもらう→rewardをrewardsにappend→学習
            # 1play → max_episode_stepsだけコイントスをして終了
            rewards = agent.play(env)
            means.append(np.mean(rewards))
        result["epsilon={}".format(e)] = means
    result["coin toss count"] = game_steps
    result = pd.DataFrame(result)
    result.set_index("coin toss count", drop=True, inplace=True)
    result.plot.line(figsize=(10, 5))
    plt.show()


main()


# -----------------------------------------------------------------------
# 環境(FrozenLake-v0)の準備
import gym
from gym.envs.registration import register
# FrozenLakeTakafumi-v0としてゲームを定義
#register(id="FrozenLakeTakafumi-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",kwargs={"is_slippery": False})

# エージェントの状態価値(Q)を可視化するための関数
# →4x4の迷路における各マスを3x3に区切り可視化する
def show_q_value(Q):
    """
    Show Q-values for FrozenLake-v0.
    To show each action's evaluation,
    a state is shown as 3 x 3 matrix like following.

       +---+---+---+---+---+---+
    5  |   | u |   |   | u |   |  u: up value
    4  | l | m | r | l | m | r |  l: left value, r: right value, m: mean value
    3  |   | d |   |   | d |   |  d: down value
       +---+---+---+---+---+---+
    2  |   | u |   |   | u |   |  u: up value
    1  | l | m | r | l | m | r |  l: left value, r: right value, m: mean value
    0  |   | d |   |   | d |   |  d: down value
       +---+---+---+---+---+---+
         0   1   2   3   4   5
    """
    env = gym.make("FrozenLakeTakafumi-v0")
    nrow = env.unwrapped.nrow # 4
    ncol = env.unwrapped.ncol # 4
    state_size = 3

    # 12 x 12
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))

    # reward_mapを埋めていく作業
    for r in range(nrow):
        for c in range(ncol):
            # s: 0 ~ 15(3*4+3)
            s = r * nrow + c
            state_exist = False

            # 左項: QがdictならTrue
            if isinstance(Q,dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True

            if state_exist:
                # (nrow - 1 - r)の変化: 3,2,1,0
                #  → _r の値の変化: 10,7,4,1
                _r = (nrow - 1 - r) * state_size + 1
                _c = (nrow - 1 - c) * state_size + 1
                # Q[s] = [x,x,x,x] 長さ4のリスト
                reward_map[_r][_c - 1] = Q[s][0] # LEFT = 0
                reward_map[_r - 1][_c] = Q[s][1] # DOWN = 1
                reward_map[_r][_c + 1] = Q[s][2] # RIGHT = 2
                reward_map[_r + 1][_c] = Q[s][3] # UP = 3
                reward_map[_r][_c] = np.mean(Q[s])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
                vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol -0.5)
    ax.set_ylim(-0.5, q_nrow -0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()

# MonteCarloAgent, QLearningAgentの親
class ELAgent(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.Q = {}
        self.reward_log = []

    def policy(self, s, actions):
        # どれでもよい
        # if random.random() < self.epsilon:
        # if np.random.rand() < self.epsilon:
        if np.random.random() < self.epsilon:
            # return random.choice(actions)
            return np.random.randint(len(actions))
        else:
            # return np.argmax(self.V)
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    # 獲得した報酬の記録
    def log(self, reward):
        self.reward_log.append(reward)

    # 記録した報酬のlogの可視化
    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            # 平均を少数第3位以上で四捨五入
            mean = np.round(np.mean(rewards), 3)
            # 標準偏差
            std = np.round(np.std(rewards), 3)
            print("エピソード{} 平均: {} (+/-: {})".format(episode, mean, std))
        # 引数episodeが与えられなかったら -> 全エピソード分のグラフを表示
        # interval=50なので、50episodeごとにrewardの平均をとりプロット
        # → 50episode全てゴールしたら1になる
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()

import math
from collections import defaultdict
class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
                render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        # 経験を貯める: 指定されたエピソード数繰り返す
        for e in range(episode_count):
            s = env.reset()
            done = False
            experience = []
            reward_sum = 0 # 1episodeで得たrewardの合計
            while not done : # エピソード終了までplay
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                reward_sum += reward
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else: # エピソードが終了したら得たrewardをlogる
                self.log(reward_sum)

            # episodeが終了したら学習: 各state, action の評価
            # i: 0から始まるインデックス、x: experienceが格納される
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                # t: iから見て何ステップ先か
                G, t = 0, 0
                for j in range(i,len(experience)):
                    # gammaのt乗(報酬割引) * time j で得た報酬
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1 # 状態sで行動aをとった回数の数え上げ
                alpha = 1 / N[s][a]

                # 今までのQ[s][a]が1-alpha分、Gがalpha分
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            # 50エピソードごとに""エピソード{} 平均: ~"とprint
            if e != 0 and e % report_interval == 0:
                # episodeを引数に指定しているのでprintされる
                self.show_reward_log(episode=e)

class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    # MonteCarloと比較して、引数にlearning_rateが増えている
    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))

        # 経験を貯める: 指定されたエピソード数繰り返す
        for e in range(episode_count):
            s = env.reset()
            done = False
            reward_sum = 0 # 1episodeで得たrewardの合計
            while not done : # エピソード終了までplay
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                reward_sum += reward

                # エピソード中に学習 → experienceも貯める必要なし
                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
            else: # エピソードが終了したら得たrewardをlogる
                self.log(reward_sum)

            # episodeが終了したら学習 → いらない

            # 50エピソードごとに""エピソード{} 平均: ~"とprint
            if e != 0 and e % report_interval == 0:
                # episodeを引数に指定しているのでprintされる
                self.show_reward_log(episode=e)

class SARSAAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    # MonteCarloと比較して、引数にlearning_rateが増えている
    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))

        # 経験を貯める: 指定されたエピソード数繰り返す
        for e in range(episode_count):
            s = env.reset()
            done = False
            reward_sum = 0 # 1episodeで得たrewardの合計
            a = self.policy(s, actions)
            while not done : # エピソード終了までplay
                if render:
                    env.render()
                n_state, reward, done, info = env.step(a)
                reward_sum += reward

                # 次にとる行動を決めておく
                n_action = self.policy(n_state, actions) # On-policy

                # エピソード中に学習 → experienceも貯める必要なし
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action # 決めておいた行動を次のstepでとる
            else: # エピソードが終了したら得たrewardをlogる
                self.log(reward_sum)

            # episodeが終了したら学習 → いらない

            # 50エピソードごとに""エピソード{} 平均: ~"とprint
            if e != 0 and e % report_interval == 0:
                # episodeを引数に指定しているのでprintされる
                self.show_reward_log(episode=e)

# train of MonteCarlo, QLearning, SARSA
print("-" * 50)
agent = MonteCarloAgent(epsilon=0.1)
env = gym.make("FrozenLakeTakafumi-v0")
agent.learn(env, episode_count=500)
show_q_value(agent.Q)
agent.show_reward_log()
print("-" * 50)
agent = QLearningAgent(epsilon=0.1)
env = gym.make("FrozenLakeTakafumi-v0")
agent.learn(env, episode_count=500)
show_q_value(agent.Q)
agent.show_reward_log()
print("-" * 50)
agent = SARSAAgent(epsilon=0.1)
env = gym.make("FrozenLakeTakafumi-v0")
agent.learn(env, episode_count=500)
show_q_value(agent.Q)
agent.show_reward_log()

agent.Q.keys()
agent.Q[4]
# -------------------------------------------------------------

class Actor(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        # 2つ目の引数1: 1個抽出
        a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        return a[0]

class Critic():

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)

class ActorCritic():

    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        actor.init_log()

        # 経験を貯める: 指定されたエピソード数繰り返す
        for e in range(episode_count):
            s = env.reset()
            done = False
            reward_sum = 0 # 1episodeで得たrewardの合計
            while not done : # エピソード終了までplay
                if render:
                    env.render()
                a = actor.policy(s) # actorの方策から行動取得
                n_state, reward, done, info = env.step(a)
                reward_sum += reward

                # gain = reward + gamma * max(self.Q[n_state])
                # criticによる評価値が使われている
                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated # td誤差
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = n_state
            else: # エピソードが終了したら得たrewardをlogる
                actor.log(reward_sum)

            # episodeが終了したら学習 → いらない

            # 50エピソードごとに""エピソード{} 平均: ~"とprint
            if e != 0 and e % report_interval == 0:
                # episodeを引数に指定しているのでprintされる
                actor.show_reward_log(episode=e)
        return actor, critic
# train of ActorCritic
print("-" * 50)
trainer = ActorCritic(Actor, Critic)
env = gym.make("FrozenLakeTakafumi-v0")
actor, critic = trainer.learn(env, episode_count=3000)
show_q_value(actor.Q)
actor.show_reward_log()
"""
sandbox
"""
actions = range(5)
random.choice(actions)
np.random.randint(4)
random.random()
np.random.rand()
np.random.random()
env = gym.make("FrozenLakeTakafumi-v0")
nrow = env.unwrapped.nrow
ncol = env.unwrapped.ncol
state_size = 3
q_nrow = nrow * state_size
q_ncol = ncol * state_size
reward_map = np.zeros((q_nrow, q_ncol))
reward_map
l = []
a = 0
while a < 10:
    l.append(a)
    a += 1
else:
    # t: iから見て何ステップ先か
    l.append(100)
l
# gammaのt乗(報酬割引) * time j で得た報酬
math.pow(3, 2) * experience[j]["reward"]
t += 1
N[s][a] += 1 # 状態sで行動aをとった回数の数え上げ
alpha = 1 / N[s][a]

# 今までのQ[s][a]が1-alpha分、Gがalpha分
self.Q[s][a] += (G alpha * - self.Q[s][a])

nrow, ncol = 3, 3
np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))
