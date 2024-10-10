import argparse


def main(args):
    print(args)
    parse_env(args)
    parse_model(args)
    train(args)


def parse_env(args):
    pass


def parse_model(args):
    pass


def train(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="VPME")
    parser.add_argument("--env", "-e", default="carla", help="env name")
    parser.add_argument("--task", choices=["task1", "task2"], default="task1")
    parser.add_argument("--weather", choices=["normal", "rain", "cloud"], default="normal")
    parser.add_argument("--model", default="VPME")
    parser.add_argument("--beta", type=float, default="0.1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
