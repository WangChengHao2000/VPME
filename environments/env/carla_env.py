import json

import carla
import numpy as np
import math
import time

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
        if self.town == "Town10HD":
            self.world = self.client.get_world()
        else:
            self.world = self.client.load_world(self.town)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.synchronous_mode = self.config["synchronous_mode"]
        self.set_synchronous_mode(self.synchronous_mode)

        self.timestep = 0

        self.ego = None
        self.spectator = None
        self.vehicle_list = list()
        self.sensor_list = list()
        self.route_waypoint_list = list()

        self.bev_camera = None
        self.collision = None

        self.current_waypoint_index = 0
        self.velocity = 0.0
        self.throttle = 0.0
        self.steer = 0.0
        self.previous_throttle = 0.0
        self.previous_steer = 0.0
        self.distance_from_center = 0.0
        self.angle = 0.0
        self.distance_cover_percent = 0.0
        self.is_collision = False

        self.total_distance = 50
        self.target_speed = 20
        self.max_speed = 25.0
        self.min_speed = 15.0
        self.max_distance_from_center = 4.5

    def reset(self):
        self.destroy()
        self.set_ego()
        self.set_vehicle()
        self.set_sensor()
        self.set_collision()
        for i in range(5):
            self.world.tick()
        self.set_waypoints()
        self.set_spectator()
        # self.debug()

    def step(self, action):
        self.timestep += 1

        velocity = self.ego.get_velocity()
        self.velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

        throttle = float((action[0] + 1) / 2)
        throttle = max(min(throttle, 1.0), 0.0)
        steer = float(action[1])
        steer = max(min(steer, 1.0), -1.0)

        self.throttle = throttle
        self.steer = steer

        # brake = 0
        # if self.ego.is_at_traffic_light():
        #     traffic_light = self.ego.get_traffic_light()
        #     if traffic_light.get_state() == carla.TrafficLightState.Red:
        #         brake = 1

        # self.ego.apply_control(carla.VehicleControl(steer=self.previous_steer * 0.5 + steer * (1 - 0.5),
        #                                             throttle=self.throttle * 0.5 + throttle * (1 - 0.5),
        #                                             brake=brake))

        self.is_collision = self.is_collision or self.collision.get_collision_event()

        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoint_list)):
            next_waypoint_index = waypoint_index + 1
            wp = self.route_waypoint_list[next_waypoint_index]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(self.ego.get_location() - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index
        current_waypoint = self.route_waypoint_list[self.current_waypoint_index]
        next_waypoint = self.route_waypoint_list[self.current_waypoint_index + 1]
        self.distance_from_center = distance_to_line(vector(current_waypoint.transform.location),
                                                     vector(next_waypoint.transform.location),
                                                     vector(self.ego.get_location()))
        fwd = vector(self.ego.get_velocity())
        wp_fwd = vector(current_waypoint.transform.rotation.get_forward_vector())
        self.angle = angle_diff(fwd, wp_fwd)

        obs = self.get_observation()
        done, success = self.get_is_done_or_success()
        reward = self.get_reward(done, success)

        if self.synchronous_mode:
            self.world.tick()
            time.sleep(0.01)
        else:
            time.sleep(0.01)

        return [obs, reward, done, success]

    def debug(self):
        debug = self.world.debug
        for i in range(self.total_distance):
            debug.draw_string(self.route_waypoint_list[i].transform.location, text=str(i), life_time=10000)

    def get_observation(self):
        obs = dict()
        if self.bev_camera is not None:
            bev_image = self.bev_camera.get_sensor_observation()
            obs["bev_image"] = bev_image

        normalized_velocity = self.velocity / self.target_speed
        normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
        normalized_angle = self.angle / np.deg2rad(90)
        obs["info"] = np.array(
            [self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])

        next_waypoint_list = list()
        for i in range(1, 6):
            next_waypoint_list.append(self.route_waypoint_list[self.current_waypoint_index + i])
        obs["next_waypoints"] = np.array(next_waypoint_list)

        obs["timestep"] = self.timestep

        return obs

    def get_is_done_or_success(self):
        done = self.is_collision or self.distance_from_center > self.max_distance_from_center or self.velocity > self.max_speed
        done = done or self.timestep >= 2500 or self.current_waypoint_index >= len(self.route_waypoint_list) - 10
        success = self.current_waypoint_index >= len(self.route_waypoint_list) - 10
        return done, success

    def get_reward(self, done, success):
        reward = 0

        alpha, beta, gamma = 1, 1, 1
        sin = math.sin(self.angle)
        cos = math.cos(self.angle)
        reward += alpha * abs(self.velocity * cos) - beta * abs(self.velocity * sin) - gamma * abs(
            self.velocity * self.distance_from_center / self.max_distance_from_center)

        self.distance_cover_percent = self.current_waypoint_index / self.total_distance
        reward *= self.distance_cover_percent

        if done and not success:
            reward -= 100
        elif success:
            reward += 100

        return reward

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
        ego_bp = self.blueprint_library.find(self.config["ego"]["model_name"])
        ego_transform = self.map.get_spawn_points()[0]
        ego_transform.location = carla.Location(**self.config["ego"]["transform"]["location"])
        ego_transform.rotation = carla.Rotation(**self.config["ego"]["transform"]["rotation"])
        self.ego = self.world.spawn_actor(ego_bp, ego_transform)
        # self.ego.set_autopilot(True)
        print("set ego done.")

    def set_waypoints(self):
        current_waypoint = self.map.get_waypoint(self.ego.get_location())
        for i in range(self.total_distance + 10):
            self.route_waypoint_list.append(current_waypoint)
            current_waypoint = current_waypoint.next(3.0)[0]

    def set_spectator(self):
        self.spectator = self.world.get_spectator()
        loc = self.ego.get_transform().location
        self.spectator.set_transform(
            carla.Transform(carla.Location(x=loc.x, y=loc.y, z=12), carla.Rotation(yaw=0, pitch=-90, roll=0)))

    def set_vehicle(self):
        vehicle_config_list = self.config["vehicle"]
        for vehicle_config in vehicle_config_list:
            vehicle_bp = self.blueprint_library.find(vehicle_config["model_name"])
            vehicle_transform = self.map.get_spawn_points()[0]
            vehicle_transform.location = carla.Location(**vehicle_config["transform"]["location"])
            vehicle_transform.rotation = carla.Rotation(**vehicle_config["transform"]["rotation"])
            vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
            self.vehicle_list.append(vehicle)

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


def angle_diff(v0, v1):
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def distance_to_line(A, B, p):
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


if __name__ == '__main__':
    env = CarlaEnv("task1")

    env.reset()
    main_done = False
    while not main_done:
        result = env.step([0, 0])
        main_done = result[2]
        print("Timestep " + str(result[0]["timestep"]) + ": reward: " + str(result[1]) + ", done: " + str(
            result[2]) + ", success: " + str(result[3]))
