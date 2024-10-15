import os
import random
import yaml
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.fft import fft
import time
import quaternion
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn

from anp.trainer import TrainerCE, TrainerMSE 
from arguments import get_config
import argparse


def get_visual_sensors_config(settings):
    # agent configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.resolution = settings["resolution"]
    rgb_sensor_cfg.far = np.iinfo(np.int32).max
    rgb_sensor_cfg.hfov = mn.Deg(settings['fov'])
    rgb_sensor_cfg.position = np.array([0, 1.5, 0])
    return [rgb_sensor_cfg]


def visual_render(sim, receiver, angles):
    rgb_panorama = []
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
    return rgb_panorama


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


def make_config_settings(scene_name="YmJkqBEsHnH", split='train', data_dir="/data/mp3d", save_dir_path='/scratch/vdj/ss/anp_depth-real-10/'):
    settings = dict(
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
        scene_obs_dir=os.path.join(save_dir_path, scene_name),
        min_distance=20,  # Half of this distance is used to select receiver location
    )
    os.makedirs(settings['scene_obs_dir'], exist_ok=True)
    return settings


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



def get_sound_intensity_from_waveform(ir_waveform, specific_acoustic_impedance_air=400):
    # Convert RIR to frequency domain
    freq_rir = fft(ir_waveform)
    # # Air density and speed of sound
    rho = 1.225  # kg/m^3
    c = 343.0    # m/s
    I = (np.abs(freq_rir) ** 2) / (rho * c)
    return np.max(I)


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


def get_decibels(ir_waveform):
    max_I = get_sound_intensity_from_waveform(ir_waveform)
    max_I = np.clip(max_I, a_min=10**-12, a_max=10**0.8) # TODO: revist this design
    return [10 * np.log10(max_I) + 120]


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


parser = argparse.ArgumentParser(description='Generate Acoustic Map Data')
parser.add_argument('--scene_name', type=str, default="YmJkqBEsHnH", help='Name of the scene')
parser.add_argument('--split', type=str, default='train', help='Data split')
parser.add_argument('--data_dir', type=str, default='/data/mp3d', help='Data directory')
parser.add_argument('--save_dir_path', type=str, default='/usr1/vdj/ss/map_acoustics', help='Save directory path')
parser.add_argument('--cm_per_pixel', type=int, default=25, help='Centimeters per pixel')
parser.add_argument('--robot', default=None, help='List of x, y and z coordinates in cm')
args = parser.parse_args()

scene_name = args.scene_name
# listener = args.listener
args.robot = [8.65, 0., -5.62] # [2.51, 0, 8.79]

split = args.split
data_dir = args.data_dir
save_dir_path = args.save_dir_path
cm_per_pixel = args.cm_per_pixel


start = time.time()
settings = make_config_settings(
    scene_name=scene_name, 
    split=split, 
    data_dir=data_dir, 
    save_dir_path=save_dir_path 
)

sim = make_sim(settings)
bounds = sim.pathfinder.get_bounds()
cm_bounds = dict(xmin=int(bounds[0][0]*100), xmax=int(bounds[1][0]*100),
                           ymin=int(bounds[0][2]*100), ymax=int(bounds[1][2]*100))

# extent = (cm_bounds['xmax'] - cm_bounds['xmin'], cm_bounds['ymax'] - cm_bounds['ymin'])

# add_audio_sensor(settings, sim, agent_id=0)
# Init robot position
# robot = sim.pathfinder.get_random_navigable_point()
# circle_centers = grid_search_for_circle_centers(sim.pathfinder.get_bounds(), num_points_per_axis=3)

# idx = random.randint(0, len(circle_centers)-1)
# for idx in range(len(circle_centers)):
    # robot_position = sim.pathfinder.get_random_navigable_point_near(circle_center=circle_centers[idx], radius=settings['min_distance'])
robot = sim.pathfinder.get_random_navigable_point() if args.robot is None else np.array(args.robot, dtype=np.float32)
save_path = os.path.join(settings['scene_obs_dir'], f"robot_at_{round(robot[0]*100)}_{round(robot[2]*100)}")
os.makedirs(save_path, exist_ok=True)
print('save_path: ', save_path)

angles = get_res_angles_for(fov=settings['visual_sensor_config']['fov'])[1]

bev_map = sim.pathfinder.get_topdown_view(meters_per_pixel=0.25, height=0)
bounds = sim.pathfinder.get_bounds()
bounds = dict(xmin=int(bounds[0][0]*100), xmax=int(bounds[1][0]*100),
                           ymin=int(bounds[0][2]*100), ymax=int(bounds[1][2]*100))
extent = [bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]]
im = plt.imshow(bev_map==False, origin="lower", cmap="binary", vmin=0, vmax=1, 
                        interpolation='none', extent=extent)
plt.savefig(os.path.join(save_path, 'bev_map.png'))
# plt.imsave(os.path.join(save_path, 'bev_map.png'), bev_map, cmap='gray')
np.save(os.path.join(save_path, 'bev_map.npy'), bev_map)

listener_positions = []
metadata = {}
gt = {}

# Visual
rgb_panorama = visual_render(sim, robot, angles)
rgb = np.concatenate(rgb_panorama, axis=1)
plt.imsave(os.path.join(save_path, f'rgb.png'), rgb)

for i in range(0, extent[1]-extent[0], cm_per_pixel):
    for j in range(0, extent[3]-extent[2], cm_per_pixel):
        print('\ni, j: ', i, j)
        position = np.array([
            (cm_bounds['xmin'] + i) / 100,
            0,
            (cm_bounds['ymin'] + j) / 100
        ], dtype=np.float32) 
        if sim.pathfinder.is_navigable(position):
            listener_positions.append(position.tolist())
        
            # Direction Distance
            distance = np.sqrt((robot[0] - position[0]) ** 2 + (robot[2] - position[2]) ** 2)
            direction = (360 - np.rad2deg(np.arctan2(robot[2] - position[2], robot[0] - position[0]))) % 360
            # # Audio
            # ir_waveform = acoustic_render(sim, listener, position, agent_id=0)
            # max_db = get_decibels(ir_waveform)
            # max_db = 0
            # Save input-output for model's pred
            metadata[f'{i}_{j}'] = [direction, distance]
            gt[f'{i}_{j}'] = position.tolist() #, max_db]
# Save metadata
with open(os.path.join(save_path, 'metadata.json'), 'w') as fo:
    json.dump(metadata, fo)

with open(os.path.join(save_path, 'gt.json'), 'w') as fo:
    json.dump(gt, fo)

with open(os.path.join(save_path, 'map_extents.json'), 'w') as fo:
    json.dump({
        "cm_bounds": cm_bounds,
        "extent_x": extent[0],
        "extent_y": extent[1],
        "cm_per_pixel": cm_per_pixel
    }, fo)

end = time.time()
print('\n Load data from: \n', save_path)
print(f"Time taken: {end - start}")