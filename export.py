import argparse
import os
import yaml

import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from pydantic import parse_obj_as

from ultralytics import YOLO
import tensorflow as tf

import shutil
from model_wrappers import Yolov8FullModel, Yolov8Keras


def export_pytorch(args, configs):
    model = Yolov8FullModel(args.model_path,
                        conf_threshold=configs['conf_threshold'],
                        iou_threshold=configs['iou_threshold'],
                        max_out_dets=configs['max_out_dets'],
                        cell_dim=configs['cell_dim']
                        )
    torch.save(model, os.path.join(os.path.dirname(args.model_path), args.model_path.split('.')[0].split('/')[-1] + '_pytorch_full_model.pt'))

    

    

def export_keras(args, configs):
    model = Yolov8Keras(args.model_path,
                    conf_threshold=configs['conf_threshold'],
                    iou_threshold=configs['iou_threshold'],
                    max_out_dets=configs['max_out_dets'],
                    cell_dim=configs['cell_dim']
                    )
    model.save(os.path.join(os.path.dirname(args.model_path), args.model_path.split('.')[0].split('/')[-1] + '_keras_full_model'))


def export_tflite(args, configs):
    model = Yolov8Keras(args.model_path,
                    conf_threshold=configs['conf_threshold'],
                    iou_threshold=configs['iou_threshold'],
                    max_out_dets=configs['max_out_dets'],
                    cell_dim=configs['cell_dim']
                    )
    # boxes, scores, classes, count = model(tmp)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(os.path.join(os.path.dirname(args.model_path), args.model_path.split('.')[0].split('/')[-1] + '_tflite_full_model.tflite')), 'wb') as f:
        f.write(tflite_model)

def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-config_path', '--config_path', type=str, help='yaml config path that contains preprocessing configs', default = 'config.yaml')
    argparser.add_argument('-model_path', '--model_path', type=str, help='pt model path', default = 'model.pt')
    argparser.add_argument("--pytorch", action='store_true', help='If added, export pytorch.')   
    argparser.add_argument("--keras", action='store_true', help='If added, export keras.')   
    argparser.add_argument("--tflite", action='store_true', help='If added, export tflite.')   
    args = argparser.parse_args()

    # read configs
    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)




    if args.pytorch:
        export_pytorch(args, configs)
    if args.keras:
        export_keras(args, configs)
    if args.tflite:
        export_tflite(args, configs)

if __name__ == '__main__':
    main()
