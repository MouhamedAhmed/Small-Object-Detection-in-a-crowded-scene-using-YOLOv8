import torch
import torch.nn as nn
import torchvision
from model_wrappers import Yolov8FullModel
from ultralytics import YOLO
import argparse

from PIL import Image, ImageDraw
import numpy as np
import copy 
import time
import itertools
from tqdm import tqdm
import random
import os
import shutil
import itertools

from predict import Predictor
from preprocess_utils import parse_json
import yaml
import json
import cv2
import pandas as pd

class NMS:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
    
    def _iou(self, box1, box2):
        dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
        dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
        if dx >=0 and dy >= 0:
            intersection = dx*dy
        else:
            intersection = 0

        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        union = area1+area2-intersection
        iou = intersection/union
        return iou

    def _get_iou_matrix(self, boxes1, boxes2):
        ious = np.zeros((len(boxes1), len(boxes2)))
        for i, p in enumerate(boxes1):
            for j, o in enumerate(boxes2):
                ious[i, j] = self._iou(p, o)

        return ious

    def _get_identical_classes_matrix(self, classes1, classes2):
        cls_compare = np.zeros((len(classes1), len(classes2)))
        for i, p in enumerate(classes1):
            for j, o in enumerate(classes2):
                cls_compare[i, j] = (p==o)

        return cls_compare

    def __call__(self, pred, gt):
        gt_boxes = gt[['x1', 'y1', 'x2', 'y2']].to_numpy()
        pred_boxes = pred[['x1', 'y1', 'x2', 'y2']].to_numpy()
        
        gt_classes = gt[['label']].to_numpy()
        pred_classes = pred[['label']].to_numpy()

        ious = self._get_iou_matrix(pred_boxes, gt_boxes)
        cls_compare = self._get_identical_classes_matrix(pred_classes, gt_classes)
        over_threshold = np.logical_and((ious > self.iou_threshold), cls_compare)

        pred_kept_indices = np.logical_not(np.any(over_threshold, axis=1))
        gt_kept_indices = np.logical_not(np.any(over_threshold, axis=0))
        pred_kept_indices = np.where(pred_kept_indices)
        gt_kept_indices = np.where(gt_kept_indices)

        pred_not_gt = pred.iloc[pred_kept_indices]
        gt_not_pred = gt.iloc[gt_kept_indices]

        return pred_not_gt, gt_not_pred




class Tester:
    def __init__(self, 
                    model_path, 
                    model_type, 
                    label_map, 
                    color_code, 
                    iou_threshold, 
                    conf_threshold, 
                    stride, 
                    cell_dim, 
                    max_out_dets, 
                    internal_boundary_filteration_thresh
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.label_map = label_map
        self.color_code = color_code
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.stride = stride
        self.cell_dim = cell_dim
        self.max_out_dets = max_out_dets
        self.internal_boundary_filteration_thresh = internal_boundary_filteration_thresh

        # predictor
        self.predictor = Predictor(
                            self.model_path, 
                            self.model_type, 
                            self.label_map,
                            self.color_code,
                            self.iou_threshold, 
                            self.conf_threshold, 
                            self.stride, 
                            self.cell_dim, 
                            self.max_out_dets, 
                            self.internal_boundary_filteration_thresh
                        )

        # initialize NMS object to make NMS between prediction and groundtruth
        self.NMS = NMS(iou_threshold)
    
    def _predict_folder(self, images_folder, gt_images):
        if os.path.isdir(images_folder):
            total_results = dict()
            vis = dict()
            images = dict()
            for image_file in tqdm(os.listdir(images_folder)):
                if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                if image_file in gt_images:
                    image_path = os.path.join(images_folder, image_file)
                    # read image
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # prediction
                    results = self.predictor.predict(image)
                    total_results[image_file] = results
                    images[image_file] = image

            return total_results, images

    def __call__(self, images_folder, annotations_json_path):
        # read gt
        anns = parse_json(annotations_json_path, images_folder, labelmap={v:k for k,v in self.label_map.items()})

        # predict 
        predictions, images = self._predict_folder(images_folder, anns.keys())

        # NMS predictions to groundtruth
        predictions_not_in_gt, gt_not_in_predictions = dict(), dict()
        for k in predictions.keys():
            predictions_not_in_gt[k], gt_not_in_predictions[k] = self.NMS(predictions[k], anns[k])

        # vis_folders
        vis_folder = images_folder + '_model_test'
        pred_folder = os.path.join(vis_folder, 'predictions')
        compare_to_gt_folder = os.path.join(vis_folder, 'compare_to_gt')
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
        if not os.path.exists(compare_to_gt_folder):
            os.mkdir(compare_to_gt_folder)

        # draw
        for image_name, image in images.items():
            pred_img = self.predictor.draw(image, predictions[image_name])
            compare_to_gt_img = self.predictor.draw(image, predictions_not_in_gt[image_name])
            compare_to_gt_img = self.predictor.draw(np.asarray(compare_to_gt_img), gt_not_in_predictions[image_name])
            # save
            pred_img.save(os.path.join(pred_folder, image_name))
            compare_to_gt_img.save(os.path.join(compare_to_gt_folder, image_name))

        #################### export metrices
        total_results = []
        
        for image_name, image in images.items():
            
            results = dict()
            # add groundtruth counts
            labels, counts = np.unique(anns[image_name]['label_name'].to_numpy(), return_counts=True)
            counts_dict = {labels[i]:counts[i] for i in range(len(labels))}
            for key in self.label_map.values():
                if key in counts_dict.keys():
                    results[key+'_gt'] = counts_dict[key]
                else:
                    results[key+'_gt'] = 0

            # add prediction counts
            labels, counts = np.unique(predictions[image_name]['label_name'].to_numpy(), return_counts=True)
            counts_dict = {labels[i]:counts[i] for i in range(len(labels))}
            for key in self.label_map.values():
                if key in counts_dict.keys():
                    results[key+'_prediction'] = counts_dict[key]
                else:
                    results[key+'_prediction'] = 0

            # add predictions_not_in_gt counts
            labels, counts = np.unique(predictions_not_in_gt[image_name]['label_name'].to_numpy(), return_counts=True)
            counts_dict = {labels[i]:counts[i] for i in range(len(labels))}
            for key in self.label_map.values():
                if key in counts_dict.keys():
                    results[key+'_prediction_not_in_gt'] = counts_dict[key]
                else:
                    results[key+'_prediction_not_in_gt'] = 0

            # add gt_not_in_predictions counts
            labels, counts = np.unique(gt_not_in_predictions[image_name]['label_name'].to_numpy(), return_counts=True)
            counts_dict = {labels[i]:counts[i] for i in range(len(labels))}
            for key in self.label_map.values():
                if key in counts_dict.keys():
                    results[key+'_gt_not_in_prediction'] = counts_dict[key]
                else:
                    results[key+'_gt_not_in_prediction'] = 0

            # add image name
            results['image'] = image_name
            total_results.append(results)

        # results
        total_results = pd.DataFrame.from_dict(total_results).fillna(0)
        images = total_results.pop('image')
        total_results.insert(0, 'image', images)

        # error
        for key in self.label_map.values():
            prediction = total_results[key+'_prediction']
            gt = total_results[key+'_gt']
            # error
            error = np.abs(prediction-gt)/gt
            total_results[key+'_error'] = error
            # true positives
            tp = prediction - total_results[key+'_prediction_not_in_gt']
            total_results[key+'_tp'] = tp
            # precision
            precision = tp/prediction
            total_results[key+'_precision'] = precision
            # recall
            recall = tp/gt
            total_results[key+'_recall'] = recall

        # get the mean 
        total_results.loc['mean'] = total_results[~total_results.isin([np.nan, np.inf, -np.inf])].mean()


        total_results.to_csv(vis_folder+'/counts.csv', index=False)


def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-config_path', '--config_path', type=str, help='yaml config path that contains preprocessing configs', default = 'config.yaml')
    argparser.add_argument('-image_folder', '--image_folder', type=str, help='path of input image folder', default = 'dataset/img')
    argparser.add_argument('-annotations_path', '--annotations_path', type=str, help='path of annotations file', default = 'dataset/whitefly.json')
    argparser.add_argument('-model_path', '--model_path', type=str, help='path of model to be tested', default = 'runs/detect/yolov8n2/weights/best.pt')
    argparser.add_argument('-model_type', '--model_type', type=str, help='type of the model to be tested (pytorch, keras, tflite)', default = 'pytorch')
    args = argparser.parse_args()

    # read configs
    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)

    # generate color_code for visualization if not exists, otherwise load it.
    if not os.path.exists("color_code.json"):
        print('Create New Color Code for Labels')
        color_code = getDistinctColors(list(cat_to_id.keys()))
        print(color_code)
        with open("color_code.json", "w") as outfile:
            json.dump(color_code, outfile)
    else:
        with open("color_code.json", 'r') as infile:
            color_code = json.load(infile)
    

    label_map = {v:k for k,v in configs['labelmap'].items()}

    # tester
    tester = Tester(
                        args.model_path, 
                        args.model_type, 
                        label_map,
                        color_code,
                        configs['iou_threshold'], 
                        configs['conf_threshold'], 
                        configs['stride'], 
                        configs['cell_dim'], 
                        configs['max_out_dets'], 
                        configs['internal_boundary_filteration_thresh']
                    )
    
    tester(args.image_folder, args.annotations_path)
    



if __name__ == '__main__':
    main()





