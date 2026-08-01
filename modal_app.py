# modal_app.py
#
# Run the SB3 Atari training on a Modal GPU instead of the local machine.
#
# One-time setup:
#   pip install modal && modal setup
#   modal secret create wandb WANDB_API_KEY=<your key>
#
# Usage (mirrors arcade/baselines/train_agent.py):
#   modal run modal_app.py --algo PPO --env ALE/Breakout-v5 --timesteps 100000 --epochs 10
#   modal run modal_app.py --algo DQN --env ALE/Pong-v5 --buffer-size 100000 --no-wandb
#
# The image uses Python 3.12 with a current SB3 stack (ale-py >= 0.10 ships the
# Atari ROMs), so use the ALE/<Game>-v5 environment ids there.

import modal

REMOTE_ROOT = '/root/game-ai'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(
        'stable-baselines3>=2.4.0',
        'gymnasium>=1.0',
        'ale-py>=0.10',
        'opencv-python-headless>=4.10',
        'tensorboard>=2.17',
        'wandb>=0.19',
    )
    .add_local_dir('arcade/baselines', remote_path=f'{REMOTE_ROOT}/baselines')
)

app = modal.App('game-ai', image=image)


@app.function(gpu='A10G', timeout=3600 * 8, secrets=[modal.Secret.from_name('wandb')])
def train_remote(algo: str, env: str, timesteps: int, epochs: int, seed: int,
                 buffer_size: int, exploration_fraction: float, no_wandb: bool):
    import sys
    sys.path.insert(0, f'{REMOTE_ROOT}/baselines')
    import train_agent

    argv = [
        '--algo', algo,
        '--env', env,
        '--timesteps', str(timesteps),
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--model-dir', f'{REMOTE_ROOT}/models/{algo}',
        '--log-dir', f'{REMOTE_ROOT}/logs',
    ]
    if buffer_size > 0:
        argv += ['--buffer-size', str(buffer_size)]
    if exploration_fraction > 0:
        argv += ['--exploration-fraction', str(exploration_fraction)]
    if no_wandb:
        argv += ['--no-wandb']

    train_agent.main(argv)


@app.local_entrypoint()
def main(algo: str = 'PPO', env: str = 'ALE/Breakout-v5', timesteps: int = 100000,
         epochs: int = 10, seed: int = 0, buffer_size: int = 0,
         exploration_fraction: float = 0.0, no_wandb: bool = False):
    train_remote.remote(algo, env, timesteps, epochs, seed, buffer_size, exploration_fraction, no_wandb)
