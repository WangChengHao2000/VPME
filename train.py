import argparse

from environments.env.carla_env import CarlaEnv
from model.ppo.ppo_agent import PPOAgent
from model.vpme.vpme_agent import VPMEAgent


def main(config):
    print(config)
    env = CarlaEnv(config["task"])
    model = parse_model(config)
    train(config, env, model)


def parse_model(config):
    model_name = config["model"]
    if model_name == "VPME":
        model = VPMEAgent()
    else:
        model = PPOAgent()
    return model


def train(config, env, model):
    episode = 0
    scores = list()
    isFinishs = list()

    train_episode = config["train_episode"]
    episode_max_step = config["train_episode"]

    while episode < train_episode:
        observation = env.reset()
        for t in range(20000):
            action_mean, action_sigma, action = model.get_action(observation, isTrain=True)
            observation, reward, done, info = env.step(action, action_mean, action_sigma)
            model.memory.observation.append(observation)
            model.memory.actions.append(action)
            model.memory.log_probs.append(logprob)
            model.memory.rewards.append(reward)
            model.memory.dones.append(done)

            if done:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="VPME")
    parser.add_argument("--env", "-e", default="carla", help="env name")
    parser.add_argument("--task", choices=["task1", "task2"], default="task1")
    parser.add_argument("--weather", choices=["normal", "rain", "cloud"], default="normal")
    parser.add_argument("--model", default="VPME")
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--train_episode", type=int, default=2000)
    parser.add_argument("--episode_max_step", type=int, default=20000)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
