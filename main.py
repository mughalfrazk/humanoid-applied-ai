import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO

# Create directories to hold models and comparelogs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    model01 = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PP0_0", seed=12, gamma=0.9, ent_coef=0.01, clip_range=0.2, learning_rate=0.001)
    model02 = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PP0_1", seed=12, gamma=0.95, ent_coef=0.5, clip_range=0.1, learning_rate=0.0009)
    model03 = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PP0_2", seed=12, gamma=0.99, ent_coef=0.1, learning_rate=0.009)
    TIMESTEPS = 500
    iters = 0
    while True:
        iters += 1

        model01.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model02.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model03.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model01.save(f"{model_dir}/{sb3_algo}_01_{TIMESTEPS * iters}")
        model02.save(f"{model_dir}/{sb3_algo}_02_{TIMESTEPS * iters}")
        model03.save(f"{model_dir}/{sb3_algo}_03_{TIMESTEPS * iters}")


def evaluate(env, path_to_model):
    model = PPO.load(path_to_model, env=env)

    obs = env.reset()[0]
    extra_steps = 100
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    envname = "Humanoid-v4"

    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--evaluate", metavar="path_to_model")
    # parser.add_argument("serial", help="Serial Number", type=int)
    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(envname, render_mode=None)
        train(gymenv, "PPO")

    if args.evaluate:
        if os.path.isfile(args.evaluate):
            gymenv = gym.make(envname, render_mode="human")
            evaluate(gymenv, path_to_model=args.evaluate)
        else:
            print(f"{args.evaluate} does not exist.")
