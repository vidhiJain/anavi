import torch
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os 
from torchvision.transforms import v2
import pandas as pd


SCENE_SPLITS = {
    'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
              'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
              'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
              'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
              'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
              'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
              '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
              'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
              'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
              '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
    'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
            'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
    'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
             'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
             'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
}


def get_ego_indices(direction):
    """Select 256x256 from the 360 degree view"""      
    dir_range = direction-45, direction+45
    indices = (128/45 *(360 - dir_range[1]) + 128),  (128/45 *(360 - dir_range[0]) + 128) 
    return indices


def get_ego_image(panorama, direction):
    l, r = get_ego_indices(direction)
    if r > 1024:
        panorama = np.concatenate([panorama]*2, axis=1)  # to handle wrap around
    return Image.fromarray(np.array(panorama)[:, int(l):int(r), :])
    

def get_sound_intensity_from_waveform(ir_waveform, specific_acoustic_impedance_air=400):
    # Convert RIR to frequency domain
    freq_rir = fft(ir_waveform)
    # # Air density and speed of sound
    rho = 1.225  # kg/m^3
    c = 343.0    # m/s
    I = (np.abs(freq_rir) ** 2) / (rho * c)
    return np.max(I)


def get_decibels(ir_waveform):
    max_I = get_sound_intensity_from_waveform(ir_waveform)
    max_I = np.clip(max_I, a_min=10**-12, a_max=10**0.8) # TODO: revist this design
    return torch.tensor([10 * np.log10(max_I) + 120])


def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError
    

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def make_data_config(split='train', dir_path = '/data/vdj/ss/anp_data_full-10/', load_images=True):
    metadata = {}
    data_path = os.path.join(dir_path, split)
    for scene in SCENE_SPLITS[split]:
        metadata_path = f'{data_path}/{scene}/metadata.json'
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        for key, val in data.items():
            metadata[f'{scene}-{key}'] = val

    audio_max_length = 50000
    return {
        'load_images': load_images,
        'dir_path': dir_path,
        'data_path': data_path,
        'split': split,
        'audio_max_length': audio_max_length,
        'metadata': metadata
    }


class AudioWaveformDataset(Dataset):
    def __init__(self, config):
        self.data_path = config['data_path']
        self.max_length = config['audio_max_length']
        self.metadata = config['metadata']   
        self.data_indices = list(self.metadata.keys()) 
        self.normalize_whole = True
        self.normalize_segment = True
        self.deterministic_eval = True
        self.use_real_imag = False
        self.hop_length = 160
        self.split = 'train'
        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or 
            # v2.Resize(size=(224, 224), antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def pad_audio(self, waveform):
        # Pad or truncate waveform to MAX_LENGTH
        if len(waveform) < self.max_length:
            padding = self.max_length - len(waveform)
            padded_waveform = np.pad(waveform, (0, padding), 'constant')
        else:
            padded_waveform = waveform[:self.max_length]
        return padded_waveform
    
    def process_audio(self, audio):
        # TODO: normalize the intensity before padding
        padded_waveform = self.pad_audio(audio)
        return padded_waveform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        seq_len = 4
        scene, idx = self.data_indices[index].split('-')
        rgb = Image.open(os.path.join(self.data_path, scene, f'{idx}-rgb.png')).convert('RGB')
        depth = Image.open(os.path.join(self.data_path, scene, f'{idx}-depth.png')).convert('RGB')

        receiver_audio = self.process_audio(wavfile.read(os.path.join(self.data_path, scene, f'{idx}-ir_receiver.wav'))[1])
        direction_distance = np.array(self.metadata[f'{scene}-{idx}'], dtype=np.float32)
        direction_distance[0] = direction_distance[0] / 360.0  # make the magnitudes similar to the direction for learning
        direction_distance[1] = direction_distance[1] / 10.0  # make the magnitudes similar to the distance for learning
        return {
            'direction_distance': torch.tensor(direction_distance).float(),
            'rgb': self.transform(rgb),
            'depth': self.transform(depth),
            'noise': receiver_audio,
        }
    
class AudioIntensityDataset(AudioWaveformDataset):
    """Simplied form of audio, just represented in highest Amplitude"""

    def process_audio(self, receiver_audio):
        receiver_audio = super().process_audio(receiver_audio)
        # waveform = self.normalize_waveform(waveform)
        # src_spec, recv_spec, src_wav, recv_wav = super().process_audio(source_audio, receiver_audio)
        return get_sound_intensity_from_waveform(receiver_audio)
    

class AudioDecibelDataset(AudioWaveformDataset):
    """Simplied form of audio, just represented in highest Amplitude"""

    def process_audio(self, receiver_audio):
        receiver_audio = super().process_audio(receiver_audio)
        return get_decibels(receiver_audio)
    

class AudioDecibelDatasetv2(Dataset):
    """Assumes max_db.csv exists in data_path. 
    This is a faster and simplified version of AudioDecibelDataset"""
    def __init__(self, config):
        self.load_images = config['load_images']
        self.dir_path = config['dir_path']
        self.data_path = config['data_path']
        self.metadata = config['metadata']   
        self.split = config['split']
        self.target_df = pd.read_csv(
            os.path.join(self.dir_path, 'max_db.csv')
        )
        self.target_df.set_index(['split', 'scene', 'idx'], inplace=True)
        self.data_indices = list(self.metadata.keys()) 

        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or 
            # v2.Resize(size=(224, 224), antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        scene, idx = self.data_indices[index].split('-')
        receiver_audio = self.target_df.loc[(self.split, scene, int(idx)), 'max_db']
        direction_distance = np.array(self.metadata[f'{scene}-{idx}'], dtype=np.float32)
        direction_distance[0] = direction_distance[0] / 360.0  # make the magnitudes similar to the direction for learning
        direction_distance[1] = direction_distance[1] / 10.0  # make the magnitudes similar to the distance for learning
        out = {
            'direction_distance': torch.tensor(direction_distance).float(),
            'noise': torch.tensor([receiver_audio]),
        }
        if self.load_images:
            rgb = Image.open(os.path.join(self.data_path, scene, f'{idx}-rgb.png')).convert('RGB')
            depth = Image.open(os.path.join(self.data_path, scene, f'{idx}-depth.png')).convert('RGB')
            out.update({
                'rgb': self.transform(rgb),
                'depth': self.transform(depth),
            })
        return out
    

class EgoViewDecibelDataset(AudioDecibelDatasetv2):
    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, index):
        scene, idx = self.data_indices[index].split('-')
        receiver_audio = self.target_df.loc[(self.split, scene, int(idx)), 'max_db']
        
        direction_distance = np.array(self.metadata[f'{scene}-{idx}'], dtype=np.float32)
        relative_direction = direction_distance[0] % 360
        direction_distance[0] = direction_distance[0] / 360.0  # make the magnitudes similar to the direction for learning
        direction_distance[1] = direction_distance[1] / 10.0  # make the magnitudes similar to the distance for learning
        out = {
            'direction_distance': torch.tensor(direction_distance).float(),
            'noise': torch.tensor([receiver_audio]),
        }
        if self.load_images:
            rgb = Image.open(os.path.join(self.data_path, scene, f'{idx}-rgb.png')).convert('RGB')
            depth = Image.open(os.path.join(self.data_path, scene, f'{idx}-depth.png')).convert('RGB')
            out.update({
                'rgb': self.transform(get_ego_image(rgb, relative_direction)),
                'depth': self.transform(get_ego_image(depth, relative_direction)),
            })
        return out


def consolidate_metadata(shard_folders, split='train'):
    consolidated_metadata = {}
    
    for shard_folder in shard_folders:
        for scene in SCENE_SPLITS[split]:
            metadata_path = os.path.join(shard_folder, split, scene, "metadata.json")
        
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    shard_metadata = json.load(f)
                    
                # Merge the shard metadata into the consolidated dictionary
                consolidated_metadata.update({f'{shard_folder}.train.{scene}.{key}': val for key, val in shard_metadata.items()}) 
    
    return consolidated_metadata


def consolidate_max_db(shard_folders):
    max_db_dfs = []
    count = 0
    for shard_folder in shard_folders:
        max_db_path = os.path.join(shard_folder, 'max_db.csv')
        
        if os.path.exists(max_db_path):
            max_db_df = pd.read_csv(max_db_path, index_col=None)
            print(f'Size of {shard_folder}: {len(max_db_df)}')
            max_db_df = max_db_df.loc[:, ~max_db_df.columns.str.contains('^Unnamed')]

            if 'shard' not in max_db_df.columns:
                max_db_df['shard'] = shard_folder.split('/')[-1]
            print(max_db_df.columns)
            assert len(max_db_df.columns) == 5, f"max_db.csv should have exactly 5 columns but there are {max_db_df.columns.tolist()}."
            max_db_dfs.append(max_db_df)
            count += len(max_db_df)
    print(f"Total number of samples: {count}")
    return pd.concat(max_db_dfs)


def make_consolidated_data_config_v3(shard_folders, split='train', load_images=True):
    metadata = consolidate_metadata(shard_folders)
    max_db_df = consolidate_max_db(shard_folders)
    audio_max_length = 50000
    return {
        'load_images': load_images,
        'split': split,
        'metadata': metadata,
        'max_db_df': max_db_df
    }


class AudioDecibelDatasetv3(Dataset):
    """Assumes max_db.csv exists in data_path for each shard. 
    This is a faster and simplified version of AudioDecibelDataset"""
    def __init__(self, config):
        self.load_images = config['load_images']
        self.split = config['split']
        self.metadata = config['metadata']   
        self.target_df = config['max_db_df']
        self.target_df.set_index(['shard', 'split', 'scene', 'idx'], inplace=True)
        self.data_indices = list(self.metadata.keys()) 

        self.transform = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or 
            # v2.Resize(size=(224, 224), antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        shard, split, scene, idx = self.data_indices[index].split('.')
        receiver_audio = self.target_df.loc[(shard.split('/')[-1], split, scene, int(idx)), 'max_db']
        direction_distance = np.array(self.metadata[f'{shard}.{split}.{scene}.{idx}'], dtype=np.float32)
        direction_distance[0] = direction_distance[0] / 360.0  # make the magnitudes similar to the direction for learning
        direction_distance[1] = direction_distance[1] / 10.0  # make the magnitudes similar to the distance for learning
        out = {
            'direction_distance': torch.tensor(direction_distance).float(),
            'noise': torch.tensor([receiver_audio]),
        }
        if self.load_images:
            try:
                rgb = Image.open(os.path.join(shard, split, scene, f'{idx}-rgb.png')).convert('RGB')
                depth = Image.open(os.path.join(shard, split, scene, f'{idx}-depth.png')).convert('RGB')
            except (FileNotFoundError, OSError):
                dummy_image = np.zeros((256, 1024, 3), dtype=np.uint8)
                rgb = Image.fromarray(dummy_image)
                depth = Image.fromarray(dummy_image)
            out.update({
                'rgb': self.transform(rgb),
                'depth': self.transform(depth),
            })
        return out
    

class EgoViewDecibelDatasetv3(AudioDecibelDatasetv3):
    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, index):
        shard, split, scene, idx = self.data_indices[index].split('.')
        receiver_audio = self.target_df.loc[(shard.split('/')[-1], split, scene, int(idx)), 'max_db']
        direction_distance = np.array(self.metadata[f'{shard}.{split}.{scene}.{idx}'], dtype=np.float32)
        relative_direction = direction_distance[0] % 360
        direction_distance[0] = direction_distance[0] / 360.0  # make the magnitudes similar to the direction for learning
        direction_distance[1] = direction_distance[1] / 10.0  # make the magnitudes similar to the distance for learning
        out = {
            'direction_distance': torch.tensor(direction_distance).float(),
            'noise': torch.tensor([receiver_audio]),
        }
        if self.load_images:
            rgb = Image.open(os.path.join(shard, split, scene, f'{idx}-rgb.png')).convert('RGB')
            depth = Image.open(os.path.join(shard, split, scene, f'{idx}-depth.png')).convert('RGB')
            out.update({
                'rgb': self.transform(get_ego_image(rgb, relative_direction)),
                'depth': self.transform(get_ego_image(depth, relative_direction)),
            })
        return out
    

if __name__=="__main__":
    # config = make_data_config('train', '/data/vdj/ss/anp_data_full-10/')
    # dataset = EgoViewDecibelDataset(config)
    # print(len(dataset))
    # print(dataset[0])

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # entry = next(iter(dataloader))
    # breakpoint()
    # print("Done")
    config = make_consolidated_data_config_v3([
        '/scratch/vdj/ss/anp_shard-30_samples-100',
        '/scratch/vdj/ss/anp_shard-2_samples-100',
        '/scratch/vdj/ss/anp_shard-27_samples-100',
        '/scratch/vdj/ss/anp_shard-7_samples-100',
        '/scratch/vdj/ss/anp_shard-24_samples-100',
        '/scratch/vdj/ss/anp_shard-8_samples-100',
        '/scratch/vdj/ss/anp_shard-3_samples-100',
        '/scratch/vdj/ss/anp_shard-6_samples-100',
        '/scratch/vdj/ss/anp_shard-14_samples-100',
        '/scratch/vdj/ss/anp_shard-16_samples-100',
        '/scratch/vdj/ss/anp_shard-26_samples-100',
        '/scratch/vdj/ss/anp_shard-32_samples-100',
        '/scratch/vdj/ss/anp_shard-17_samples-100',
        '/scratch/vdj/ss/anp_shard-15_samples-100'
    ], split='train', load_images=True)

    dataset = AudioDecibelDatasetv3(config)
    print(len(dataset))
    breakpoint()
    print("done")