import argparse
import os
import sys

from chainer import optimizers
import gym
from gym import spaces
import numpy as np

import chainerrl
from chainerrl.agents.dqn import DQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer

from q_function import QFunction, DuelingQFunction, QFunctionCartPole
from erlam import ERLAM


def main():
    import logging
    logging.basicConfig(level=20)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10**3)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
    parser.add_argument('--dueling', action='store_true', default=False,
                        help='use dueling dqn')
    parser.add_argument('--k', type=int, default=20, help='number of K')
    parser.add_argument('--lambdas', type=float, default=0.1)
    parser.add_argument('--dim', type=int, default=4, help='dimension for random projection')
    parser.add_argument('--no_erlam', action='store_true', default=False, help='dqn de training')
    parser.add_argument('--start_size', type=int, default=5000, help='start to use graph')
    parser.add_argument('--graph_capacity', type=int, default=5000, help='max size of graph')
    parser.add_argument('--k_neighbor', type=int, default=3, help='k nearest neighbor')
    args = parser.parse_args()

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if ((args.render_eval and test) or
                (args.render_train and not test)):
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space)
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.2
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = action_space.n
        q_func = QFunctionCartPole(
            obs_size, n_actions,)
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if args.minibatch_size is None:
        args.minibatch_size = 32
    if args.prioritized_replay:
        betasteps = (args.steps - args.replay_start_size) \
            // args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            rbuf_capacity, betasteps=betasteps)
    else:
        rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)


    Agent = ERLAM
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99, k=args.k,
                  lambdas=args.lambdas, dim=args.dim, start_size=args.start_size,
                  capacity=args.graph_capacity,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=False,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  phi=lambda x: x,)

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            train_max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
