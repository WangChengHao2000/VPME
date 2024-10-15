import carla
from abc import ABC, abstractmethod
import sys

import environments.config.base_config as base_config


class EnvBase(ABC):
    def __init__(self):
        self.client = None

        # noinspection PyBroadException
        try:
            print("connect...")
            self.client = carla.Client(base_config.HOST, base_config.PORT)
            self.client.set_timeout(base_config.TIMEOUT)
            print("connect done.")
        except Exception:
            print("connect error")
            sys.exit(0)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass
