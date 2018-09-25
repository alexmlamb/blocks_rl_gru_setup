from rainy import agent, Config, net
from rainy.agent import Agent
from rainy.env_ext import Atari, EnvExt
from rainy.explore import EpsGreedy, LinearCooler
from typing import Optional
from torch.optim import RMSprop


def run_agent(ag: Agent, eval_env: Optional[EnvExt] = None):
    max_steps = ag.config.max_steps
    turn = 0
    rewards_sum = 0
    while True:
        if max_steps and ag.total_steps > max_steps:
            break
        if turn != 0 and turn % 100 == 0:
            print('turn: {}, total_steps: {}, rewards: {}'.format(
                turn,
                ag.total_steps,
                rewards_sum
            ))
            rewards_sum = 0
        if turn != 0 and turn % 1000 == 0:
            print('eval: {}'.format(ag.eval_episode(env=eval_env)))
        if turn != 0 and turn % 10000 == 0:
            ag.save("saved-example.rainy")
        rewards_sum += ag.episode()
        turn += 1
    ag.save("saved-example.rainy")


def run(train: bool = True):
    c = Config()
    c.max_steps = 100000
    c.double_q = True
    a = agent.DqnAgent(c)
    if train:
        run_agent(a)
    else:
        a.load("saved-example.rainy")
        print('eval: {}'.format(a.eval_episode()))


def run_atari(train: bool = True):
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=True))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    )
    c.set_explorer(
        lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, int(1e6)), net)
    )
    c.double_q = True
    c.set_value_net(net.value_net.dqn_conv)
    c.replay_size = int(1e6)
    c.batch_size = 32
    c.train_start = 50000
    c.sync_freq = 10000
    c.max_steps = int(2e7)
    a = agent.DqnAgent(c)
    eval_env = Atari('Breakout', frame_stack=True, episode_life=False)
    if train:
        run_agent(a, eval_env=eval_env)
    else:
        a.load("saved-example.rainy")
        print('eval: {}'.format(a.eval_episode(env=eval_env)))


if __name__ == '__main__':
    run_atari(train=False)
