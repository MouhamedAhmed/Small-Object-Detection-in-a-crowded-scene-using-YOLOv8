import cv2      
import torch  
import torch.nn as nn
from tensorflow import keras
import tensorflow as tf
from model_wrappers import Yolov8FullModel
import itertools
import numpy as np 
import torchvision
import argparse
import yaml
from preprocess_utils import draw_bbs, plot_color_legend
import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm 

class NMS_Tiles:
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

    def __call__(self, tile1_preds, tile2_preds):
        # iou matrix between each pair of predictions from the 2 tiles
        ious = self._get_iou_matrix(tile1_preds['boxes'], tile2_preds['boxes'])
        # check if each pair of predictions is of the same class
        cls_compare = self._get_identical_classes_matrix(tile1_preds['classes'], tile2_preds['classes'])
        # matrix that contains true if the pair of predictions are overlapping and of the same class
        over_threshold = np.logical_and((ious > self.iou_threshold), cls_compare)
        # check if tile 1 is the containing the higher score for each pair
        tile1_higher_score = tile1_preds['scores'].reshape(-1, 1) > tile2_preds['scores'].reshape(1, -1)
        # tile1 kept predictions after NMS, the ones that do not overlap with tile2 predictions 
        tile2_removed = np.any(np.logical_and(over_threshold, tile1_higher_score), axis=0)
        tile1_removed = np.any(np.logical_and(over_threshold, np.logical_not(tile1_higher_score)), axis=1)
        # select predictions after NMS
        tile1_preds['boxes'] = np.array([tile1_preds['boxes'][i] for i in range(len(tile1_preds['boxes'])) if not tile1_removed[i]])
        tile1_preds['classes'] = np.array([tile1_preds['classes'][i] for i in range(len(tile1_preds['classes'])) if not tile1_removed[i]])
        tile1_preds['scores'] = np.array([tile1_preds['scores'][i] for i in range(len(tile1_preds['scores'])) if not tile1_removed[i]])
        tile2_preds['boxes'] = np.array([tile2_preds['boxes'][i] for i in range(len(tile2_preds['boxes'])) if not tile2_removed[i]])
        tile2_preds['classes'] = np.array([tile2_preds['classes'][i] for i in range(len(tile2_preds['classes'])) if not tile2_removed[i]])
        tile2_preds['scores'] = np.array([tile2_preds['scores'][i] for i in range(len(tile2_preds['scores'])) if not tile2_removed[i]])

        return tile1_preds, tile2_preds




class Predictor:
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
                    internal_boundary_filteration_thresh,
                    is_tile
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
        self.is_tile = is_tile

        # load model 
        self._read_model()

        # initialize NMS object to make NMS between prediction and groundtruth
        self.NMS = NMS_Tiles(iou_threshold)
    
    def _read_model(self):
        # load model
        if self.model_type == 'pytorch':
            self.model = Yolov8FullModel(self.model_path,
                        conf_threshold=self.conf_threshold,
                        iou_threshold=self.iou_threshold,
                        max_out_dets=self.max_out_dets,
                        cell_dim=self.cell_dim
                        )
        elif self.model_type == 'keras':
            self.model = keras.models.load_model(self.model_path)
        elif self.model_type =='tflite':
            self.model = tf.lite.Interpreter(model_path = self.model_path)
            self.model.allocate_tensors()
        else:
            self.model = None



    def _tile(self, image):
        height, width = image.shape[:2]

        xs = list(range(0, width, self.stride))
        for i in range(len(xs)):
            if xs[i] + self.cell_dim > width:
                xs[i] = width - self.cell_dim
                break
        xs = xs[:i+1]

        ys = list(range(0, height, self.stride))
        for i in range(len(ys)):
            if ys[i] + self.cell_dim > height:
                ys[i] = height - self.cell_dim
                break
        ys = ys[:i+1]

        starts = list(itertools.product(xs, ys))

        tiles =  [image[j1:j1+self.cell_dim, i1:i1+self.cell_dim] for i1, j1 in starts]

        return tiles, starts



    def _predict_tile(self, tile):
        if self.model_type == 'pytorch':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            tile = torch.Tensor((np.expand_dims(tile, 0) / 255.0).transpose((0,3,1,2))).half().to(device)
            print(tile.shape)
            print(self.model(tile))
            bboxes, scores, classes, _ = self.model(tile)
            bboxes = bboxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            classes = classes.cpu().detach().numpy()

        elif self.model_type == 'keras':
            tile = (np.expand_dims(tile, 0) / 255.0)
            tile = tf.convert_to_tensor(tile, dtype=tf.float32) 
            bboxes, scores, classes, _ = self.model.predict(tile, verbose=0)
        elif self.model_type =='tflite':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            tile = (np.expand_dims(np.array(tile), 0) / 255).astype('float32')
            self.model.set_tensor(input_details[0]['index'], tile)
            self.model.invoke()
            out = [self.model.get_tensor(od['index']) for od in output_details]
            _, bboxes, classes, scores = out
        else:
            bboxes, scores, classes = None, None, 
                
        return {
            'boxes':bboxes,
            'scores':scores,
            'classes':classes
        }

    def _stitch(self, image, results, start_coords):
        h, w = image.shape[:2]

        # delete objects close to internal boundaries
        internal_boundary_filteration_ratio = self.internal_boundary_filteration_thresh/self.cell_dim
        max_object_size = (self.cell_dim - self.stride - 2*self.internal_boundary_filteration_thresh)/self.cell_dim

        for i in range(len(results)):
            
            tile_x1, tile_y1, tile_x2, tile_y2 = start_coords[i][0], start_coords[i][1], start_coords[i][0]+self.cell_dim, start_coords[i][1]+self.cell_dim
            left_internal, top_internal, right_internal, bottom_internal = (tile_x1 != 0), (tile_y1 != 0), (tile_x2 != w), (tile_y2 != h)
            
            to_be_removed_idx = []
            for box_idx, box in enumerate(results[i]['boxes']):
                x1, y1, x2, y2 = box
                # filter if internal boundary and obejct is so close to boundary (Note: Do NOT filter if close to left or top and object is relatively big) OR score is low
                if (left_internal and (x1 < internal_boundary_filteration_ratio) and (x2-x1 < max_object_size)) or \
                   (right_internal and (x2 > 1-internal_boundary_filteration_ratio)) or \
                   (top_internal and (y1 < internal_boundary_filteration_ratio) and (y2-y1 < max_object_size)) or \
                   (bottom_internal and (y2 > 1-internal_boundary_filteration_ratio)) or \
                   results[i]['scores'][box_idx] < self.conf_threshold:
                    to_be_removed_idx.append(box_idx)
            for removed_idx in sorted(np.unique(to_be_removed_idx), reverse=True):
                results[i]['boxes'] = np.delete(results[i]['boxes'], removed_idx, axis=0)
                results[i]['scores'] = np.delete(results[i]['scores'], removed_idx, axis=0)
                results[i]['classes'] = np.delete(results[i]['classes'], removed_idx, axis=0)
        # revert back from tile coords to image coords 
        for i in range(len(results)):
            start = [
                    start_coords[i][1],
                    start_coords[i][0],
                    start_coords[i][1],
                    start_coords[i][0]
                     ]
            results[i]['boxes'] = np.array([
                        ((box*self.cell_dim)+start)/np.array([w,h,w,h])
                        for box in results[i]['boxes']
                    ])
            
        # NMS
        for i in range(10,len(results)-1):
            for j in range(i+1, len(results)):
                tile1_coords = np.array(start_coords[i])
                tile2_coords = np.array(start_coords[j])
                if np.all(tile2_coords <= tile1_coords + self.cell_dim):
                    results[i], results[j] = self.NMS(results[i], results[j])
        # stitch
        stitched_result = dict()
        stitched_result['boxes'], stitched_result['classes'], stitched_result['scores'] = [], [], []
        for result in results:
            stitched_result['boxes'] += result['boxes'].tolist()
            stitched_result['scores'] += result['scores'].tolist()
            stitched_result['classes'] += result['classes'].tolist()
        stitched_result['boxes'] = np.array(stitched_result['boxes'])
        stitched_result['scores'] = np.array(stitched_result['scores'])
        stitched_result['classes'] = np.array(stitched_result['classes'])
        return stitched_result


    def draw(self, image, results):
        # convert cv2 image to pil
        image = Image.fromarray(image) 
        # plot bbs on images using the previously-generated color code 
        image = draw_bbs(image, results, self.color_code) 
        # plot color code on image
        image = plot_color_legend(image, self.color_code)
        return image




    def predict(self, image):
        if self.is_tile:
            # tile the image
            tiles, start_coords = self._tile(image)

            # run model
            results = []
            for i, tile in enumerate(tiles):
                results.append(self._predict_tile(tile))

            # NMS and stitch
            results = self._stitch(image, results, start_coords)
        else:
            size = (self.cell_dim, self.cell_dim)
            image = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
            results = self._predict_tile(image)


            

        # convert to dataframe
        label_names = [self.label_map[i] for i in results['classes']]
        if len(results['boxes']):
            results = pd.DataFrame(list(zip(results['classes'], label_names, results['scores'],  results['boxes'][:,0], results['boxes'][:,1], results['boxes'][:,2], results['boxes'][:,3])),
                    columns =['label', 'label_name', 'score', 'x1', 'y1', 'x2', 'y2'])
        else:
            results = pd.DataFrame(columns=['label', 'label_name', 'score', 'x1', 'y1', 'x2', 'y2'])

        return results


def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-config_path', '--config_path', type=str, help='yaml config path that contains preprocessing configs', default = 'config.yaml')
    argparser.add_argument('-image_path', '--image_path', type=str, help='path of input image', default = 'dataset/img/01FNBG8AAVF31MR05362PGKJ3Z.jpeg')
    argparser.add_argument('-model_path', '--model_path', type=str, help='path of model to be tested', default = 'runs/detect/yolov8n2/weights/best.pt')
    argparser.add_argument('-model_type', '--model_type', type=str, help='type of the model to be tested (pytorch, keras, tflite)', default = 'pytorch')
    argparser.add_argument('-color_code', '--color_code', type=str, help='path of color code json file', default = 'color_code.json')
    args = argparser.parse_args()

    # read configs
    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)

    # read color_code for visualization.
    with open(args.color_code, 'r') as infile:
        color_code = json.load(infile)
    

    label_map = {k[1]:k[0] for k in configs['labelmap'].items()}

    # predictor
    predictor = Predictor(
                            args.model_path, 
                            args.model_type, 
                            label_map,
                            color_code,
                            configs['iou_threshold'], 
                            configs['conf_threshold'], 
                            configs['stride'], 
                            configs['cell_dim'], 
                            configs['max_out_dets'], 
                            configs['internal_boundary_filteration_thresh'],
                            configs['const_tiles']
                        )
    
    # predict on a folder
    if os.path.isdir(args.image_path):
        vis_folder = args.image_path + '_model_predictions'
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
        total_results = []
        for image_file in tqdm(os.listdir(args.image_path)):
            dst_path = os.path.join(vis_folder, image_file)
            image_path = os.path.join(args.image_path, image_file)
            if not image_path.endswith(('.jpg', '.png', '.jpeg')):
                continue
            # read image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # prediction
            results = predictor.predict(image)
            # visualize
            vis_image = predictor.draw(image, results)
            # save visualized image
            vis_image.save(dst_path)
            labels, counts = np.unique(results['label_name'].to_numpy(), return_counts=True)
            results = {labels[i]:counts[i] for i in range(len(labels))}
            results['image'] = image_file
            total_results.append(results)
    
        total_results = pd.DataFrame.from_dict(total_results).fillna(0)
        total_results.insert(0, 'image', total_results.pop('image'))
        total_results.to_csv(vis_folder+'.csv', index=False)

    # predict on single image
    else:
        if not args.image_path.endswith(('.jpg', '.png', '.jpeg')):
            print('ERROR: image_path is not an image')
        vis_folder = os.path.dirname(args.image_path) + '_model_predictions'
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
        dst_path = os.path.join(vis_folder, args.image_path.split('/')[-1])
        # read image
        image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # prediction
        results = predictor.predict(image)
        # visualize
        vis_image = predictor.draw(image, results)
        # save visualized image
        vis_image.save(dst_path)
        # print result
        labels, counts = np.unique(results['label_name'].to_numpy(), return_counts=True)
        print(label_map)
        print({labels[i]:counts[i] for i in range(len(labels))})




if __name__ == '__main__':
    main()


# image_path = 'dataset/img/01FNBG8AAVF31MR05362PGKJ3Z.jpeg'
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# model_path = 'runs/detect/yolov8n2/weights/best.pt'
# model_type = 'pytorch'
# # model_path = 'runs/detect/yolov8n2/weights/best_keras_full_model'
# # model_type = 'keras'
# # model_path = 'runs/detect/yolov8n2/weights/best_tflite_full_model.tflite'
# # model_type = 'tflite'

# iou_threshold = 0.3
# conf_threshold = 0.25
# stride = 500
# cell_dim = 640
# max_out_dets = 1000

# predictor = Predictor(model_path, model_type, iou_threshold, conf_threshold, stride, cell_dim, max_out_dets)
# predictor.predict(image)
