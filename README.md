# Small-Object-Detection-in-a-crowded-scene-using-YOLOv8

## Preprocess Dataset
```bash
python3.10 preprocess.py --config_path <CONFIG_YAML_PATH>
```
where `CONFIG_YAML_PATH` is the config file path containing all configurations for the preprocessing process, default is `config.yaml`.

## Train YOLOv8n Model
```bash
python3.10 train.py --config_path <CONFIG_YAML_PATH>
```
where `CONFIG_YAML_PATH` is the config file path containing all configurations for the training process, default is `config.yaml`.

## Export Model to TFLite and Keras
```bash
python3.10 export.py --config_path <CONFIG_YAML_PATH> --model_path <PYTORCH_MODEL_PATH> <MODEL_TYPE>
```
where 
- `CONFIG_YAML_PATH` is the config file path containing all configurations for the exporting process, default is `config.yaml`.
- `PYTORCH_MODEL_PATH` is the trained pytorch model path, typically in `runs/detect/yolov8n/weights/best.pt`, default is `model.pt`.
- `MODEL_TYPE` is the model types to be exported, one of (--keras, --tflite) or both.

## Predict on Image or Folder of Images  



## Test Model

