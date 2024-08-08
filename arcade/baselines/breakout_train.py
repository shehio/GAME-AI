from helpers import create_directory_if_not_exists, create_atari_environment, train
from stable_baselines3 import PPO
import time

env_name = 'Breakout-v4'
log_directory = 'logs'
model_directory = 'models/PPO'

env = create_atari_environment(env_name, render=False)
create_directory_if_not_exists(model_directory)
create_directory_if_not_exists(log_directory)

# Use 'CnnPolicy' for image-based observations like Atari Breakout
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_directory)

start_time = time.time()

timesteps = 10000
epochs = 1000
model = train(model, timesteps, epochs, env_name, model_directory)
model.save('final_ppo_breakout')

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time / 60} minutes")