# train_agent.py
#
# CLI wrapper around helpers.train so the same training loop can be driven by
# hand, by "wandb sweep" agents (see sweeps/), or by Modal (see modal_app.py).
#
#   python train_agent.py --algo PPO --env Breakout-v4 --timesteps 100000 --epochs 5
#   python train_agent.py --algo DQN --env Pong-v4 --buffer-size 100000 --no-wandb

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from stable_baselines3 import DQN, PPO

from helpers import create_atari_environment, create_directory_if_not_exists, train, wandb_enabled

ALGOS = {'PPO': PPO, 'DQN': DQN}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train an SB3 agent on an Atari environment.')
    parser.add_argument('--algo', choices=sorted(ALGOS), default='PPO')
    parser.add_argument('--env', default='Breakout-v4')
    parser.add_argument('--timesteps', type=int, default=10000, help='timesteps per epoch')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', '--learning_rate', type=float, default=None)
    parser.add_argument('--buffer-size', '--buffer_size', type=int, default=None, help='DQN only')
    parser.add_argument('--exploration-fraction', '--exploration_fraction', type=float, default=None, help='DQN only')
    parser.add_argument('--eval-episodes', '--eval_episodes', type=int, default=5, help='0 skips the final evaluation')
    parser.add_argument('--torch-threads', '--torch_threads', type=int, default=None)
    parser.add_argument('--model-dir', '--model_dir', default=None)
    parser.add_argument('--log-dir', '--log_dir', default=os.path.join(SCRIPT_DIR, 'logs'))
    parser.add_argument('--tag', action='append', default=None, help='wandb tag, repeatable')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb tracking')
    return parser.parse_args(argv)


def build_model(args, env):
    kwargs = {'seed': args.seed}
    if args.learning_rate is not None:
        kwargs['learning_rate'] = args.learning_rate
    if args.algo == 'DQN':
        if args.buffer_size is not None:
            kwargs['buffer_size'] = args.buffer_size
        if args.exploration_fraction is not None:
            kwargs['exploration_fraction'] = args.exploration_fraction
    return ALGOS[args.algo]('CnnPolicy', env, verbose=1, tensorboard_log=args.log_dir, **kwargs)


def main(argv=None):
    args = parse_args(argv)

    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)

    model_directory = args.model_dir or os.path.join(SCRIPT_DIR, 'models', args.algo)
    create_directory_if_not_exists(model_directory)
    create_directory_if_not_exists(args.log_dir)

    env = create_atari_environment(args.env, render=False)
    model = build_model(args, env)
    model = train(model, args.timesteps, args.epochs, args.env, model_directory, tags=args.tag)
    model.save(f"{model_directory}/final_{args.algo.lower()}_{args.env.replace('/', '_')}")

    if args.eval_episodes > 0:
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
        print(f"Evaluation over {args.eval_episodes} episodes: {mean_reward} +/- {std_reward}")
        if wandb_enabled():
            import wandb
            if wandb.run is not None:
                wandb.run.summary['eval/mean_reward'] = mean_reward
                wandb.run.summary['eval/std_reward'] = std_reward
                wandb.run.summary['eval/episodes'] = args.eval_episodes

    env.close()


if __name__ == '__main__':
    main()
