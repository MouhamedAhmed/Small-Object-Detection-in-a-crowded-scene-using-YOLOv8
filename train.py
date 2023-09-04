import yaml
import pathlib
import argparse
from ultralytics import YOLO
import os

def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-config_path', '--config_path', type=str, help='yaml config path', default = 'config.yaml')
    argparser.add_argument("--resume", action='store_true', help='If added, resume training.')   
    args = argparser.parse_args()

    # read configs
    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)

    # resolve dataset path
    if configs['const_tiles']:
        data_folder = 'yolo_dataset_const_tiles'
    else:
        data_folder = 'yolo_dataset_random_tiles'
    dataset_path = os.path.join(configs['dataset_path'], data_folder)

    # create train data config
    dict_file = dict(
        path = str(pathlib.Path(__file__).parent.resolve()) + '/',
        train = os.path.join(dataset_path, 'train'),
        test = os.path.join(dataset_path, 'test'),
        val = os.path.join(dataset_path, 'valid'),
        nc = len(configs['labelmap']),
        names = list(configs['labelmap'].keys())
    )

    # save train data config
    with open('yolov8-data-config.yaml', 'w') as file:
        documents = yaml.safe_dump(dict_file, file)
    
    # Load the model.
    if configs['model_path'] is None:
        model_path = configs['model_version'] + '.pt'
    else:
        model_path = configs['model_path']
    model = YOLO(model_path)
    
    # Training.
    results = model.train(
        data='yolov8-data-config.yaml',
        imgsz=configs['cell_dim'],
        epochs=configs['epochs'],
        batch=configs['batch_size'],
        max_det=configs['max_det'],
        name=configs['model_version'],
        resume=args.resume
    )

if __name__ == '__main__':
    main()

