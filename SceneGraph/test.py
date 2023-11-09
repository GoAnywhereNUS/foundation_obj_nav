
import habitat_sim
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "/home/zhanxin/Desktop/L3MVN/data/scene_datasets/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
backend_cfg.scene_dataset_config_file = "/home/zhanxin/Desktop/L3MVN/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
}
agent_cfg.sensor_specifications = [sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
    for region in scene.regions:
        print(region)
        try:
            print(f"Region id:{region.id}, category:{region.category}")
            print(f" center:{region.aabb.center}, dims:{region.aabb.sizes}")
        except:
            print('error')
    # for obj in scene.objects:
    #     print(f"category:{obj.category.name()}")
    #     print(f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}")
    # count += 1
    # if count >= limit_output:
    #     return None
scene = sim.semantic_scene
# import pdb
# pdb.set_trace()
print_scene_recur(scene)
