import os
import quaternion
import habitat_sim.sim
from scipy.io import wavfile
import multiprocessing
import math
import os
import json
import time
import magnum as mn
import numpy as np

from habitat_sim.utils.common import quat_from_angle_axis
import habitat_sim
from scipy.io import wavfile

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the Agg backend for rendering to files
matplotlib.rcParams["figure.dpi"] = 600

from scene_splits import SCENE_SPLITS
from audio_utils import get_decibels
import argparse

def make_config_settings(scene_name="YmJkqBEsHnH", split='train', data_dir="/data/mp3d", save_dir_path='/scratch/vdj/ss/anp_depth-real-10/'):
    settings = dict(
        split=split,
        scene_name=scene_name,
        scene_id=f"{data_dir}/{scene_name}/{scene_name}.glb", 
        scene_dataset_config_file=f"{data_dir}/mp3d.scene_dataset_config.json",
        navmesh=f"{data_dir}/{scene_name}/{scene_name}.navmesh",
        mp3d_material_config_file=f"{data_dir}/mp3d_material_config.json", 
        visual_sensor_config=dict(
            resolution=(256, 256), 
            fov=90,
            # fov=60,
        ),
        sampling_rate=44100,
        num_per_scene=int(save_dir_path.split("-")[-1]),
        scene_obs_dir=os.path.join(save_dir_path, split, scene_name),
        min_distance=20,  # Half of this distance is used to select receiver location
    )
    os.makedirs(settings['scene_obs_dir'], exist_ok=True)
    return settings


def get_visual_sensors_config(settings):
    # agent configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.resolution = settings["resolution"]
    rgb_sensor_cfg.far = np.iinfo(np.int32).max
    rgb_sensor_cfg.hfov = mn.Deg(settings['fov'])
    rgb_sensor_cfg.position = np.array([0, 1.5, 0])

    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = 'depth_camera'
    depth_sensor_cfg.resolution = settings["resolution"]
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.hfov = mn.Deg(settings['fov'])
    depth_sensor_cfg.position = np.array([0, 1.5, 0])

    semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
    semantic_sensor_cfg.uuid = "semantic_camera"
    semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_cfg.resolution = settings["resolution"]
    semantic_sensor_cfg.hfov = mn.Deg(settings['fov'])
    semantic_sensor_cfg.position = np.array([0, 1.5, 0])

    return [rgb_sensor_cfg, depth_sensor_cfg, semantic_sensor_cfg]


def add_audio_sensor(settings, sim, agent_id):
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = True
    audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
    audio_sensor_spec.channelLayout.channelCount = 1
    # audio sensor location set with respect to the agent
    audio_sensor_spec.position = [0.0, 1.5, 0.0]  # audio sensor has a height of 1.5m
    audio_sensor_spec.acousticsConfig.sampleRate = settings['sampling_rate']
    
    sim.add_sensor(audio_sensor_spec, agent_id)
    audio_sensor = sim.get_agent(agent_id)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON(settings['mp3d_material_config_file'])
    return audio_sensor


def visual_render(sim, receiver, angles):
    rgb_panorama = []
    depth_panorama = []
    semantic_panorama = []
    for angle in angles:
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = receiver
        new_state.rotation = quat_from_angle_axis(math.radians(angle), np.array([0, 1, 0]))
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        print(f'Agent set at {agent.get_state()}')

        observation = sim.get_sensor_observations()
        rgb_panorama.append(observation["rgba_camera"][..., :3])
        depth_panorama.append(normalize_depth(observation['depth_camera']))
        semantic_panorama.append(observation['semantic_camera'])

    return rgb_panorama, depth_panorama, semantic_panorama


def acoustic_render(sim, receiver, source, agent_id=0):
    # Returns the IR at the receiver agent location
    audio_sensor = sim.get_agent(agent_id)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(source)

    agent = sim.get_agent(agent_id)
    new_state = agent.get_state()
    new_state.position = receiver
    # new_state.rotation = quat_from_angle_axis(0, np.array([0, 1, 0]))
    new_state.sensor_states = {}
    agent.set_state(new_state, True)
    observation = sim.get_sensor_observations()
    return np.array(observation['audio_sensor'])


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def make_sim(settings):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = settings['scene_id']
    backend_cfg.scene_dataset_config_file = settings['scene_dataset_config_file']
    
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False
    # Add agents
    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = get_visual_sensors_config(settings['visual_sensor_config'])
    cfg = habitat_sim.Configuration(backend_cfg, [agent_config])
    sim = habitat_sim.Simulator(cfg)
    sim.pathfinder.load_nav_mesh(settings['navmesh'])
    
    return sim


def get_res_angles_for(fov):
    if fov == 20:
        resolution = (384, 64)
        angles = [170, 150, 130, 110, 90, 70, 50, 30, 10, 350, 330, 310, 290, 270, 250, 230, 210, 190]
    elif fov == 30:
        resolution = (384, 128)
        angles = [0, 330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30]
    elif fov == 60:
        resolution = (256, 128)
        angles = [0, 300, 240, 180, 120, 60]
    elif fov == 90:
        resolution = (256, 256)
        angles = [0, 270, 180, 90]
    else:
        raise ValueError

    return resolution, angles
    

def process_seg_mask(sem, unique_classes=100):
    breakpoint()
    H, W = sem.shape
    K = unique_classes
    one_hot_encoded = np.zeros((H, W, K), dtype=np.bool_)
    for i, cls in enumerate(unique_classes):
        one_hot_encoded[:, :, i] = (sem == cls)
    return one_hot_encoded


def grid_search_for_circle_centers(bounds, num_points_per_axis=10):
    min_vector, max_vector = bounds
    x_min, y_min = min_vector[0], min_vector[2]
    x_max, y_max = max_vector[0], max_vector[2]
    x_points = np.linspace(x_min, x_max, num_points_per_axis)
    y_points = np.linspace(y_min, y_max, num_points_per_axis)
    circle_centers = []
    for x in x_points:
        for y in y_points:
            circle_centers.append(np.array([x, 0.17, y], dtype=np.float32))
    return circle_centers


def create_data(settings):
    receivers = []
    sources = []
    metadata = {}
    gt = {}

    angles = get_res_angles_for(fov=settings['visual_sensor_config']['fov'])[1]
    sim = make_sim(settings)
    grid_occupancy = sim.pathfinder.get_topdown_view(meters_per_pixel=0.01, height=0.0)
    plt.clf()
    plt.matshow(grid_occupancy)
    title = f'Map for {settings["scene_name"]}'
    plt.title(title)
    plt.savefig(os.path.join(settings['scene_obs_dir'], f'map-{settings["scene_name"]}.png'))

    num_points_per_axis=int(math.sqrt(settings['num_per_scene']))
    circle_centers = grid_search_for_circle_centers(sim.pathfinder.get_bounds(), num_points_per_axis)
    for i in range(settings['num_per_scene']//2): # saving source and receiver interchangeably twice.
        source = sim.pathfinder.get_random_navigable_point_near(circle_center=circle_centers[i % (num_points_per_axis**2)], radius=settings['min_distance'], max_tries=100)
        receiver = sim.pathfinder.get_random_navigable_point_near(circle_center=source, radius=settings['min_distance']/2.0, max_tries=100) 
        
        # TODO: check the distances where it makes sense for interesting training data
        if not (np.sqrt((source[0] - receiver[0]) ** 2 + (source[2] - receiver[2]) ** 2) < settings['min_distance'] and \
                abs(source[1] - receiver[1]) < 2):
            # Sample a random source and receiver at min_distance/2
            source = sim.pathfinder.get_random_navigable_point()
            receiver = sim.pathfinder.get_random_navigable_point_near(circle_center=source, radius=settings['min_distance']/2.0, max_tries=100)
            
        receivers.append(receiver)
        sources.append(source)
        distance = np.sqrt((source[0] - receiver[0]) ** 2 + (source[2] - receiver[2]) ** 2)
        direction = (360 - np.rad2deg(np.arctan2(source[2] - receiver[2], source[0] - receiver[0]))) % 360
        metadata[2*i] = (direction, distance)
        gt[2*i] = {
            'map_id': settings['scene_id'],
            'source': source.tolist(),
            'receiver': receiver.tolist(),
        }

        receivers.append(source)
        sources.append(receiver)
        distance = np.sqrt((source[0] - receiver[0]) ** 2 + (source[2] - receiver[2]) ** 2)
        direction = (360 - direction) % 360
        metadata[2*i + 1] = (direction, distance)
        gt[2*i + 1] = {
            'map_id': settings['scene_id'],
            'source': receiver.tolist(),
            'receiver': source.tolist(),
        }

    for i, source in enumerate(sources):
        # capture panorama at source location
        rgb, depth, semantic = visual_render(sim, source, angles)
        # # DEBUG: save all the raw data - too bulky file! 
        # rgbs.append(rgb)
        # depths.append(depth)
        # semantics.append(semantic)
        # np.savez(os.path.join(settings['scene_obs_dir'], f'{i}-combined.npz'), rgb=rgb, depth=depth, semantic=semantic)
        
        # # # For debugging visualization!
        rgb = np.concatenate(rgb, axis=1)
        depth = np.concatenate(depth, axis=1)
        semantic = np.concatenate(semantic, axis=1)
        
        # rgb = np.stack(rgb, axis=0)
        # rgb = rgb[:, :, :, :3].transpose(0, 3, 1, 2)
        # depth = np.stack(depth, axis=0)
        # depth = np.expand_dims(depth, axis=1)
        # semantic = np.stack(semantic, axis=0)
        # semantic = np.expand_dims(semantic, axis=1)
        # one_hot_semantic = process_seg_mask(semantic)
        # combined_obs = np.concatenate([rgb, depth, semantic], axis=-1)
        
        plt.imsave(os.path.join(settings['scene_obs_dir'], f'{i}-rgb.png'), rgb)
        plt.imsave(os.path.join(settings['scene_obs_dir'], f'{i}-depth.png'), depth)
        # # OR save a matshow of the depth
        # plt.clf()
        # plt.matshow(depth)
        # plt.savefig(os.path.join(settings['scene_obs_dir'], f'{i}-depth.png'))

        # # capture IR at source location
        # ir = acoustic_render(sim, source+np.array([0.0,1.0,0.0]), source)
        # ir_source_obs.append(ir)
    # sim.close()

    # sim = make_sim(settings)
    add_audio_sensor(settings, sim, agent_id=0) # Add audio sensor to the agent
    max_dbs = []
    for i, (receiver, source) in enumerate(zip(receivers, sources)):
        ir = acoustic_render(sim, receiver, source, agent_id=0)
        max_dbs.append(get_decibels(ir))
        wavfile.write(os.path.join(settings['scene_obs_dir'], f'{i}-ir_receiver.wav'), settings['sampling_rate'], ir[0])
        # np.save(os.path.join(settings['scene_obs_dir'], f'{i}-ir.npy'), ir)
        # ir_receiver_obs.append(ir)
    sim.close()



    # Saving stuff
    with open(os.path.join(settings['scene_obs_dir'], 'gt.json'), 'w') as fo:
        json.dump(gt, fo)

    with open(os.path.join(settings['scene_obs_dir'], 'metadata.json'), 'w') as fo:
        json.dump(metadata, fo)

    # # Saving preprocessing later for training data 
    # max_db_dict = {f"{settings['split']}-{settings['scene_name']}-{idx}": max_dbs[idx] for idx in range(len(max_dbs))}
    # with open(os.path.join(settings['scene_obs_dir'], 'max_db.json'), 'w') as fo:
    #     json.dump(max_db_dict, fo)

    # np.savez(
    #     os.path.join(settings['scene_obs_dir'], f'{i}-combined.npz'), 
    #     rgbs=rgbs, 
    #     depths=depths, 
    #     semantics=semantics, 
    #     max_dbs=max_dbs, 
    #     receivers=receivers, 
    #     sources=sources, 
    #     metadata=metadata, 
    #     map_id=settings['scene_id']
    # )

    # # DEBUG: Save the IRs and images 
    # for i, (ir_source, ir_receiver) in enumerate(zip(ir_source_obs, ir_receiver_obs)):
        # wavfile.write(os.path.join(settings['scene_obs_dir'], f'{i}-ir_source.wav'), 44100, ir_source[0])
        # wavfile.write(os.path.join(settings['scene_obs_dir'], f'{i}-ir_receiver.wav'), 44100, ir_receiver[0])
        # plt.imsave(os.path.join(settings['scene_obs_dir'], f'{i}-rgb.png'), rgb)
        # plt.imsave(os.path.join(settings['scene_obs_dir'], f'{i}-depth.png'), depth)
        # plt.imsave(os.path.join(settings['scene_obs_dir'], f'{i}-sem.png'), sem)


def process_scene_split(args):
    scene, split, data_dir, save_dir_path = args
    settings = make_config_settings(scene_name=scene, split=split, data_dir=data_dir, save_dir_path=save_dir_path)
    create_data(settings)


def get_safe_cpu_count(use_cpus=None, leave_cpus=8):
    # Get the number of CPU cores available on the system
    # Adjust `leave_cpus` value based on your system's capacity
    if use_cpus is not None:
        return use_cpus
    available_cpus = multiprocessing.cpu_count()
    safe_cpus = max(1, available_cpus - leave_cpus)  
    return safe_cpus


def get_files_under_threshold(directory, threshold):
    files_under_threshold = []
    os.chdir(directory)
    dirs = os.listdir()
    for name in dirs:
        file_size = os.path.getsize(name)
        if file_size <= threshold:
            files_under_threshold.append(name)
    print('files_under_threshold: ', files_under_threshold)
    return files_under_threshold


def main(args):
    data_dir = args.data_dir
    shard_index = args.shard_index
    num_samples = args.num_samples
    save_dir_path = os.path.join(
        args.save_dir, f'anp_shard-{shard_index}_samples-{num_samples}'
    )
    os.makedirs(save_dir_path)
    first_run = True
    threshold_size = 0  # if 0 then all folders to be generated; 4096 if some folder contain only map. 
    args_list = []
    for split in ['train']: # , 'val', 'test']:
        if not first_run:
            files_under_threshold = get_files_under_threshold(save_dir_path + '/' + split, threshold_size)
            print(files_under_threshold)
        for scene in SCENE_SPLITS[split]:
            if not first_run:
                if scene in files_under_threshold:
                    args_list.append((scene, split, data_dir, save_dir_path))
            else:
                args_list.append((scene, split, data_dir, save_dir_path))
    print('Args_list', args_list)
    
    with multiprocessing.Pool(processes=get_safe_cpu_count(use_cpus=args.use_cpus)) as pool:
        pool.map(process_scene_split, args_list)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--shard_index', type=int, default=0, help='Index of the shard')
    parser.add_argument('--data_dir', type=str, default='/data/mp3d', help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='/scratch/vdj/ss/', help='Path to the save directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scene')
    parser.add_argument('--use_cpus', type=int, default=None, help='Number of CPU cores to use')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    