# Acoustic Noise Data Generation 

# Setup
Follow Sound-spaces Installation instructions. Specifically, install the following: 

1. habitat-sim (RLRAudioPropagationUpdate branch)
1. habitat-lab (0.2.2) 
1. mp3d dataset
1. `data/mp3d_material_config.json`

# Run 
to generate training data for ANP.
```
python mp3d_generate.py
``` 

* Specify the location (like a `/scratch` folder on a cluster) for generating files. 
* This generates 140-160Gb of data. 
* Adjust the number of samples per scene and the number of scenes for fewer samples to be generated. 

# Generate data 
to visualize ANP for Fixed Robot  
```
python generate_acoustic_map_data.py  
```

and for Fixed Listener
```
python generate_listener_map_data.py
```
