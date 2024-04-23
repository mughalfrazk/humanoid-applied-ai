import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG

# Create directories to hold models and comparelogs
model_dir = "models"
log_dir = "comparelogs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    if sb3_algo == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif sb3_algo == "TD3":
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif sb3_algo == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif sb3_algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif sb3_algo == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        print("Invalid SB3 algo")
        return

    TIMESTEPS = 500
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")


def evaluate(env, sb3_algo, path_to_model):
    global model
    if sb3_algo == "SAC":
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == "TD3":
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == "A2C":
        model = A2C.load(path_to_model, env=env)
    elif sb3_algo == "PPO":
        model = PPO.load(path_to_model, env=env)
    elif sb3_algo == "DDPG":
        model = DDPG.load(path_to_model, env=env)
    else:
        print("Invalid SB3 algo")
        return

    obs = env.reset()[0]
    done = False
    extra_steps = 100
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == "__main__":
    envname = "Humanoid-v4"

    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("sb3_algo", help="StableBaseline3 RL algorithm i.e. SAC, TD3, A2C")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--evaluate", metavar="path_to_model")
    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(envname, render_mode=None)
        train(gymenv, args.sb3_algo)

    if args.evaluate:
        if os.path.isfile(args.evaluate):
            gymenv = gym.make(envname, render_mode="human")
            evaluate(gymenv, args.sb3_algo, path_to_model=args.evaluate)
        else:
            print(f"{args.evaluate} does not exist.")
