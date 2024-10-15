import carla
import numpy as np


class Camera(object):
    def __init__(self, world, ego, position):
        self.observation = np.zeros((300, 400, 3))

        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()

        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "400")
        camera_bp.set_attribute("image_size_y", "300")
        camera_bp.set_attribute("fov", "100")

        assert position in ["front", "left", "right", "rear"]
        self.position = position

        if position == "front":
            camera_transform = carla.Transform(carla.Location(x=1.3, y=2.3, z=2.4),
                                               carla.Rotation(roll=0, pitch=0, yaw=0))
        elif position == "left":
            camera_transform = carla.Transform(carla.Location(x=1.3, y=2.3, z=2.4),
                                               carla.Rotation(roll=0, pitch=0, yaw=-60))
        elif position == "right":
            camera_transform = carla.Transform(carla.Location(x=1.3, y=2.3, z=2.4),
                                               carla.Rotation(roll=0, pitch=0, yaw=60))
        else:
            camera_transform = carla.Transform(carla.Location(x=-1.3, y=2.3, z=2.4),
                                               carla.Rotation(roll=0, pitch=0, yaw=-180))

        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        self.camera.listen(lambda image: self.__parse_image__(image))

    def __parse_image__(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.observation = array

    def get_sensor_observation(self):
        return self.observation

    def destroy(self):
        self.camera.destroy()


class SemanticCamera(object):
    def __init__(self, world, ego):
        self.observation = np.zeros((75, 100, 3))

        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()

        camera_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", "100")
        camera_bp.set_attribute("image_size_y", "75")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(carla.Location(x=4.0, y=0, z=16.0),
                                           carla.Rotation(pitch=-90.0, yaw=0, roll=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        self.camera.listen(lambda image: self.__parse_image__(image))

    def __parse_image__(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.array(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.observation = array

    def get_sensor_observation(self):
        return self.observation

    def destroy(self):
        self.camera.destroy()


class Collision(object):
    def __init__(self, world, ego):
        self.collision_event = False

        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()

        collision_bp = self.blueprint_library.find("sensor.other.collision")
        collision_transform = carla.Transform(carla.Location(x=1.3, z=0.5))
        self.collision = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=ego
        )
        self.collision.listen(lambda event: self.__parse_collision__())

    def __parse_collision__(self):
        self.collision_event = True

    def get_collision_event(self):
        return self.collision_event

    def destroy(self):
        self.collision.destroy()
