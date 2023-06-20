# Small-Object-Detection-in-a-crowded-scene-using-YOLOv8

## Preprocess Dataset
```bash
python3.10 preprocess.py --config_path <CONFIG_YAML_PATH>
```
where `CONFIG_YAML_PATH` is the config file path containing all configurations for the preprocessing process, default is "config.yaml".

## Train YOLOv8n Model
```bash
python3.10 train.py --config_path <CONFIG_YAML_PATH>
```
where `CONFIG_YAML_PATH` is the config file path containing all configurations for the training process, default is "config.yaml".

## Export Model to TFLite and Keras


## Test Model

