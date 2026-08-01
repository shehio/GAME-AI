# helpers.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os
import sys
import time

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None

# Newer ale-py (>= 0.9) needs explicit registration with gymnasium >= 1.0;
# older stacks self-register through the plugin entry point.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass

WANDB_ENTITY = 'shehio'
WANDB_PROJECT = 'game-ai'


def wandb_enabled():
    if wandb is None or '--no-wandb' in sys.argv:
        return False
    return os.environ.get('WANDB_MODE') != 'disabled'


def build_wandb_config(model, env_name, timesteps, epochs):
    config = {
        'algo': type(model).__name__,
        'env_name': env_name,
        'policy': type(model.policy).__name__,
        'timesteps_per_epoch': timesteps,
        'epochs': epochs,
        'total_timesteps': timesteps * epochs,
        'seed': model.seed,
        'algo_params': {},
    }
    for key in ('learning_rate', 'gamma', 'batch_size', 'n_steps', 'n_epochs',
                'ent_coef', 'buffer_size', 'learning_starts', 'exploration_fraction',
                'exploration_initial_eps', 'exploration_final_eps', 'target_update_interval', 'tau'):
        value = getattr(model, key, None)
        if isinstance(value, (int, float, str)):
            config['algo_params'][key] = value
    return config

def create_atari_environment(name: str, render: bool, render_fps = 60):
    render_mode = 'human' if render else None
    env = gym.make(name, render_mode=render_mode)
    env = AtariWrapper(env)
    if render:
        env.metadata['render_fps'] = render_fps
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def create_environment(name: str, render: bool, render_fps = 60):
    render_mode = 'human' if render else None
    env = gym.make(name, render_mode=render_mode)
    if render:
        env.metadata['render_fps'] = render_fps
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def train(model, timesteps, epochs, tensorboard_log_name, model_directory, tags=None):
    run = None
    callback = None
    if wandb_enabled():
        try:
            run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                config=build_wandb_config(model, tensorboard_log_name, timesteps, epochs),
                tags=tags,
                sync_tensorboard=True,  # forwards the SB3 tensorboard scalars (rollout/*, train/*, time/*)
            )
            callback = WandbCallback(verbose=2)
        except Exception as e:
            print(f"wandb unavailable, training without tracking: {e}")
            run = None

    start_time = time.time()
    for i in range(epochs):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=tensorboard_log_name, callback=callback)
        model.save(f"{model_directory}/{timesteps * i}")
        if run is not None:
            run.log({'epoch': i + 1, 'elapsed_minutes': (time.time() - start_time) / 60})

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60} minutes")
    if run is not None:
        run.summary['elapsed_minutes'] = elapsed_time / 60
    return model

def render_model(env, model, timestep: int, frame_skip: int = 10):
    obs = env.reset()
    for i in range(timestep):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i % frame_skip == 0:
            env.render()
    env.close()

def create_directory_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
