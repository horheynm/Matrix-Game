# Setup 

Download the repo
`git clone https://github.com/horheynm/Matrix-Game.git`

Go to the Matrix-Game-2 directory
`cd /Matrix-Game/Matrix-Game-2`

Generate a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

run the setup script (using virtualenv)
`bash setup.sh`


# Running infernce benchmarks / profiling

To run inference using the same input args as the repo's baseline, run
`python3 inference.py`

To apply optimizations add the `--optimizations 1` to add the VAE decode optimization and `--optimizations 2` for the denoise, `--optimizations 1,2` for both


To run the profiling, run the following

For profiling denoise, context update and decode: add the `--profile_focus all`. You can choose from `(all|decode|denoise|context)`



---
# Reproducibility

## Metrics

Baseline
```
python inference.py 
```

Decode optimization
```
python inference.py --optimization 1
```

Denoise optimization
```
python inference.py --optimization 2
```

Best numbers
```
python inference.py --optimization 1,2
```

## Profiling

Decode optimization with profiling
```
python inference.py --config_path configs/inference_yaml/inference_universal.yaml --img_path demo_images/universal/0000.png --num_output_frames 150 --output_folder outputs/ --pretrained_model_path /home/mac_local/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0 --bench --torch_profile --warmup_blocks 2 --max_blocks 6 --profile_focus all --profile_timesteps 0 --torch_profile_dir prof_runs/matrix_decode_opt1 --optimization 1
 ```

Denoise optimization with profiling
```
python inference.py --config_path configs/inference_yaml/inference_universal.yaml --img_path demo_images/universal/0000.png --num_output_frames 150 --output_folder outputs/ --pretrained_model_path /home/mac_local/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0 --bench --torch_profile --warmup_blocks 2 --max_blocks 6 --profile_focus all --profile_timesteps 0 --torch_profile_dir prof_runs/matrix_decode_opt2 --optimization 2
 ```

Both optimizations with profiling
```
python inference.py --config_path configs/inference_yaml/inference_universal.yaml --img_path demo_images/universal/0000.png --num_output_frames 150 --output_folder outputs/ --pretrained_model_path /home/mac_local/Matrix-Game/Matrix-Game-2/Matrix-Game-2.0 --bench --torch_profile --warmup_blocks 2 --max_blocks 6 --profile_focus all --profile_timesteps 0 --torch_profile_dir prof_runs/matrix_decode_opt12 --optimization 1,2
 ```