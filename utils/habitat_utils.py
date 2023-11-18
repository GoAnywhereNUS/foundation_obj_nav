import numpy as np
import habitat_sim


default_habitat_sim_settings = {
    "scene": "scenes/scene_datasets/habitat-test-scenes/skokloster-castle.glb",  # Scene path
    "sensor_height": 1.3,  # Height of sensors in meters, relative to the agent
    "width": 640,  # Spatial resolution of the observations
    "height": 480,
    "turn_degrees": 10.0,
}


def setup_quad_360_cam_config(settings):
    left_rgb_sensor = habitat_sim.CameraSensorSpec()
    left_rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    left_rgb_sensor.uuid = "left_rgb"
    left_rgb_sensor.resolution = [settings["height"], settings["width"]]
    left_rgb_sensor.hfov = 90
    left_rgb_sensor.position = [0.0, settings["sensor_height"], 0.0]
    left_rgb_sensor.orientation = [0, np.pi/2, 0]

    forward_rgb_sensor = habitat_sim.CameraSensorSpec()
    forward_rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    forward_rgb_sensor.uuid = "forward_rgb"
    forward_rgb_sensor.resolution = [settings["height"], settings["width"]]
    forward_rgb_sensor.hfov = 90
    forward_rgb_sensor.position = [0.0, settings["sensor_height"], 0.0]
    forward_rgb_sensor.orientation = [0., 0., 0.]

    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    right_rgb_sensor.uuid = "right_rgb"
    right_rgb_sensor.resolution = [settings["height"], settings["width"]]
    right_rgb_sensor.hfov = 90
    right_rgb_sensor.position = [0.0, settings["sensor_height"], 0.0]
    right_rgb_sensor.orientation = [0., -np.pi/2, 0.]

    rear_rgb_sensor = habitat_sim.CameraSensorSpec()
    rear_rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rear_rgb_sensor.uuid = "rear_rgb"
    rear_rgb_sensor.resolution = [settings["height"], settings["width"]]
    rear_rgb_sensor.hfov = 90
    rear_rgb_sensor.position = [0.0, settings["sensor_height"], 0.0]
    rear_rgb_sensor.orientation = [0., -np.pi, 0.]

    return [
        left_rgb_sensor,
        forward_rgb_sensor,
        right_rgb_sensor,
        rear_rgb_sensor
    ]


def setup_sim_config(settings=default_habitat_sim_settings):
    # Configure simulator
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # Configure agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = setup_quad_360_cam_config(settings)
    agent_cfg.action_space["turn_left"].actuation.amount = settings["turn_degrees"]
    agent_cfg.action_space["turn_right"].actuation.amount = settings["turn_degrees"]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])