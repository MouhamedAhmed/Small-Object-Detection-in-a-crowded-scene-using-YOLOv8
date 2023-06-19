
import torch  
import torch.nn as nn
import tensorflow as tf
import torchvision
from ultralytics import YOLO
import shutil


class Yolov8FullModel(nn.Module):
    def __init__(self, model_path:str,
                       conf_threshold:float,
                       iou_threshold:float,
                       max_out_dets:int,
                       cell_dim:int
                 ):
        super().__init__()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path)['model'].to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_out_dets = max_out_dets
        self.cell_dim = cell_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # prediction
        yolo_preds = self.model(image)[0]

        # get number of classes
        n_classes = yolo_preds.shape[1] - 4

        # split to boxes, classes, scores
        scores = torch.max(yolo_preds[:, 4:4+n_classes], 1)[0][0]
        classes = torch.argmax(yolo_preds[:, 4:4+n_classes], 1)[0]
        boxes = yolo_preds[:, 0:4][0]

        # convert boxes from (center_x, center_y, w, h) to (top, left, bottom, right) 
        boxes = torch.transpose(boxes, 0, 1)
        xc = boxes[:, 0]
        yc = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = xc - w//2
        x2 = xc + w//2
        y1 = yc - h//2
        y2 = yc + h//2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # filter on confidence
        selected_indices = scores >= self.conf_threshold
        boxes, scores, classes = boxes[selected_indices], scores[selected_indices], classes[selected_indices]
  
        # NMS
        selected_indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_threshold)
        boxes, scores, classes = boxes[selected_indices], scores[selected_indices], classes[selected_indices]

        # get boxes with highest scores of max 'max_out_dets' boxes
        selected_indices = torch.topk(scores, min(self.max_out_dets, scores.shape[0])).indices
        boxes, scores, classes = boxes[selected_indices], scores[selected_indices], classes[selected_indices]

        # normalize boxes
        boxes /= self.cell_dim
        
        # avoid problem of object repitition
        if scores.shape[0] > 1:
            if torch.abs(scores[-1]-scores[-2]) < 1e-6:
                scores *= ((scores-scores[-1])>1e-5)

        
        # get count of objects
        count = (scores>1e-5).sum()

        return boxes, scores, classes, count


class Yolov8Keras(tf.keras.models.Model):
    def __init__(self, model_path:str,
                       conf_threshold:float,
                       iou_threshold:float,
                       max_out_dets:int,
                       cell_dim:int
                ):
        super(Yolov8Keras, self).__init__()

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_out_dets = max_out_dets
        self.cell_dim = cell_dim

        # load ultralytics pt model
        model = YOLO(model_path)
        model.fuse()  
        model.info(verbose=True)  

        # export saved_model
        model.export(format='saved_model', nms=True) 

        # load saved_model and delete it 
        self.model = tf.saved_model.load(model_path.split('.')[0] + '_saved_model')
        shutil.rmtree(model_path.split('.')[0] + '_saved_model')

        # self(tf.Tensor(np.ones((1, self.cell_dim, self.cell_dim, 3)).astype('float32')))
        self(tf.zeros(
                (1, self.cell_dim, self.cell_dim, 3),
                dtype=tf.dtypes.float32)
            )

    def call(self, inputs):
        # prediction
        yolo_preds = self.model(inputs)

        # get number of classes
        n_classes = yolo_preds.shape[1] - 4

        # split to boxes, classes, scores
        scores = tf.math.reduce_max(yolo_preds[:, 4:4+n_classes], 1)[0]
        classes = tf.argmax(yolo_preds[:, 4:4+n_classes], 1)[0]
        classes = tf.cast(classes, tf.float32)
        boxes = yolo_preds[:, 0:4][0]

        # convert boxes from (center_x, center_y, w, h) to (top, left, bottom, right) 
        boxes = tf.transpose(boxes)
        xc = boxes[:, 0]
        yc = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = xc - w//2
        x2 = xc + w//2
        y1 = yc - h//2
        y2 = yc + h//2
        boxes = tf.stack([x1, y1, x2, y2], axis=1)

        # # filter on confidence
        # selected_indices = scores >= self.conf_threshold
        # boxes = tf.gather(boxes, selected_indices)
        # scores = tf.gather(scores, selected_indices)
        # classes = tf.gather(classes, selected_indices)        

        # NMS
        selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=self.max_out_dets, iou_threshold=self.iou_threshold, score_threshold=self.conf_threshold)
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        classes = tf.gather(classes, selected_indices)

        # normalize boxes
        boxes /= self.cell_dim

        # avoid problem of object repitition
        if scores.shape[0]:
            scores = tf.math.multiply(scores, tf.cast((scores-scores[-1]>1e-5), tf.float32))

        # get count of objects
        count = tf.math.reduce_sum(tf.cast(scores>1e-5, 'int32'))

        return boxes, scores, classes, count


