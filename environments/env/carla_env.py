import carla
import numpy as np
import cv2
import json
from environments.env.env_base import EnvBase
from environments.env.sensor import SemanticCamera, Collision


class CarlaEnv(EnvBase):
    def __init__(self, task):
        super().__init__()
        taskFileName = "environments/config/" + task + ".json"
        with open(taskFileName, 'r') as taskFile:
            self.config = json.load(taskFile)
        print(self.config)

        self.town = self.config["map"]
        self.world = self.client.load_world(self.town)
        self.blueprint_library = self.world.get_blueprint_library()

        self.set_synchronous_mode(True)

        self.ego = None
        self.spectator = None
        self.vehicle_list = list()
        self.sensor_list = list()

        self.bev_camera = None
        self.collision = None

        self.throttle = 0.0
        self.steer = 0.0
        self.previous_throttle = 0
        self.previous_steer = 0

    def reset(self):
        self.destroy()
        self.set_ego()
        self.set_spectator()
        self.set_vehicle()
        self.set_sensor()
        self.set_collision()

    def step(self, action):
        throttle = float((action[0] + 1) / 2)
        throttle = max(min(throttle, 1.0), 0.0)
        steer = float(action[1])
        steer = max(min(steer, 1.0), -1.0)

        self.throttle = throttle
        self.steer = steer

        # self.action_mean = action_mean
        # self.action_mean[0] = max(min((self.action_mean[0] + 1) / 2, 1.0), 0.0)
        # self.action_sigma = action_sigma

        self.vehicle.apply_control(
            carla.VehicleControl(steer=self.previous_steer * 0.5 + steer * (1 - 0.5),
                                 throttle=self.throttle * 0.5 + throttle * (1 - 0.5)))

        self.world.tick()
        self.set_spectator()
        self.get_sensor()

    def get_sensor(self):
        if self.bev_camera is not None:
            bev_image = self.bev_camera.get_sensor_observation()
            cv2.imwrite("./checkpoints/images/bev.jpg", bev_image)

    def set_sensor(self):
        print("set sensor...")
        self.bev_camera = SemanticCamera(self.world, self.ego)
        self.sensor_list.append(self.bev_camera)
        print("set sensor done.")

    def set_collision(self):
        print("set collision...")
        self.collision = Collision(self.world, self.ego)
        self.sensor_list.append(self.collision)
        print("set collision done.")

    def set_ego(self):
        print("set ego...")
        vehicle_bp = self.blueprint_library.find(self.config["ego"]["model_name"])
        spawn_point = np.random.choice(self.world.get_map().get_spawn_points())
        self.ego = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego.set_autopilot(True)
        print("set ego done.")

    def set_spectator(self):
        self.spectator = self.world.get_spectator()
        loc = self.ego.get_transform().location
        self.spectator.set_transform(
            carla.Transform(carla.Location(x=loc.x, y=loc.y, z=30), carla.Rotation(yaw=0, pitch=-90, roll=0)))

    def set_vehicle(self):
        vehicle_list = self.config["vehicle"]
        for vehicle in vehicle_list:
            pass
            # TODO
            # self.vehicle_list.append()

    def set_synchronous_mode(self, mode):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = mode
        self.world.apply_settings(settings)

    def destroy(self):
        for sensor in self.sensor_list:
            sensor.destroy()
        self.sensor_list.clear()

        for vehicle in self.vehicle_list:
            vehicle.destroy()
        self.vehicle_list.clear()

        if self.ego is not None:
            self.ego.destroy()
            self.ego = None

    def __del__(self):
        self.set_synchronous_mode(False)
        print("destroy...")
        self.destroy()
        print("destroy done.")


if __name__ == '__main__':
    env = CarlaEnv("task1")

    env.reset()
    while True:
        env.step(None)
