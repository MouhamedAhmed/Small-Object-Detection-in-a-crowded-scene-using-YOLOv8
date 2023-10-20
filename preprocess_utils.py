import os
from sklearn.model_selection import train_test_split
import shutil
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from imgaug import augmenters as iaa
import albumentations as A
import json
from PIL import Image, ImageDraw, ImageFont
import random
from shapely.geometry import Polygon



def HSVToRGB(h, s, v): 
    '''
    a function that converts hsv color to rgb color
    '''
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(names): 
    '''
    a function that generates a set of visually-separable set of colors, one for each class
    '''
    base_colors = [
        (255,0,0),          # red
        (0,0,255),          # blue
        (0,0,0),            # black
        (255,255,255),      # white
        (255,0,255),        # magenta
        (0,128,128),        # teal
        (255,255,0),        # yellow
        (0,255,255),        # cyan
        (0,0,128),          # navy
        (128,0,0),          # maroon
    ]

    n = len(names)

    if n <= len(base_colors):
      colors = {names[value]: base_colors[value] for value in range(n)}
    else:
      huePartition = 3.0 / (n + 1) 
      colors = {names[value]: HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(n)}

    return colors



def plot_color_legend(img, color_code):
  draw = ImageDraw.Draw(img)
  x1, x2 = 10, 150
  y1 = 60

  font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
  font = ImageFont.truetype(font_path, size=35)

  for label, color in color_code.items():
    y2 = y1 + 50
    draw.rectangle(((x1, y1), (x2, y2)), fill=tuple(color))
    draw.text((x1, y1-40), label, font=font)
    y1 += 100
  return img
  

def draw_bbs(img, df, color_code):
  w, h = img.size
  draw = ImageDraw.Draw(img)
  # draw boxes
  for _, row in df.iterrows():
    # create rectangle image
    shape = [
            (int(row['x1']*w),
              int(row['y1']*h)),
              
            (int(row['x2']*w), 
              int(row['y2']*h))
            ]
    color = tuple(color_code[row['label_name']])
    draw.rectangle(shape, outline=color, width=2)
    if 'score' in df.columns:
      draw.text(
        (shape[0][0], shape[0][1]-10), 
        str(np.round(row['score'], 3)),
        fill=color+(255,)  
      )

  return img




def parse_json(json_path, image_folder, labelmap=None, visualize=False, vis_folder=None):
  print('Parse Json File.')
  #### read json file
  with open(json_path) as json_file:
      data = json.load(json_file)

  #### parse json file
  # get the categories dictionaries
  id_to_cat = {i['id']:i['name'] for i in data['categories']}
  cat_to_id = {i['name']:i['id'] for i in data['categories']}
  # get image names and ids
  img_ids = {p['id']:p['file_name'] for p in data['images']}

  # initialize empty lists for each image
  bboxes = {img_ids[p['image_id']]:[] for p in data['annotations']}
  classes = {img_ids[p['image_id']]:[] for p in data['annotations']}

  # parse annotations and add them to corresponding image list in the dictionaries
  for p in data['annotations']:
    bboxes[img_ids[p['image_id']]].append(p['bbox'])
    classes[img_ids[p['image_id']]].append(p['category_id'])
  
  # if visualization will be done, create random but visually-separable color code and save it
  if not os.path.exists("color_code.json"):
    print('Create New Color Code for Labels')
    color_code = getDistinctColors(list(cat_to_id.keys()))
    print(color_code)
    with open("color_code.json", "w") as outfile:
      json.dump(color_code, outfile)
  else:
    with open("color_code.json", 'r') as infile:
      color_code = json.load(infile)

  # create dataframe per image
  dfs = dict()

  if visualize == True and vis_folder is not None:
    if os.path.exists(vis_folder):
      shutil.rmtree(vis_folder)
    os.mkdir(vis_folder)

  for img_name in tqdm(list(bboxes.keys())):
    if not img_name.endswith(('.jpg', '.png', 'jpeg')):
      continue

    # get width and height
    cv2_img = cv2.imread(os.path.join(image_folder, img_name))
    h, w = cv2_img.shape[:2]

    # get bboxes in the format of (x1, y1, w, h), no ratios, actual pixels
    bboxes[img_name] = np.array(bboxes[img_name])

    # convert bboxes to the format of (x1, y1, x2, y2) (NORMALIZED)
    x1 = (bboxes[img_name][:, 0]) / w
    y1 = (bboxes[img_name][:, 1]) / h
    x2 = (bboxes[img_name][:, 0] + bboxes[img_name][:, 2]) / w
    y2 = (bboxes[img_name][:, 1] + bboxes[img_name][:, 3]) / h
    image_classes = classes[img_name]
    image_cats = [id_to_cat[i] for i in image_classes]

    df = pd.DataFrame(list(zip(image_classes, image_cats, x1, y1, x2, y2)),
                  columns =['label', 'label_name', 'x1', 'y1', 'x2', 'y2'])
    
    #### Visulize groundtruth
    if visualize:
      # convert cv2 image to pil
      img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)) 
      # plot bbs on images using the previously-generated color code 
      img = draw_bbs(img, df, color_code) 
      # plot color code on image
      img = plot_color_legend(img, color_code)
      # save the plotted images in a folder
      img.save(os.path.join(vis_folder, img_name))


    # keep required data for training only and update the labels based on the label map
    if labelmap is not None:
      df = df[df['label_name'].isin(list(labelmap.keys()))]
      for label_name, label in labelmap.items():
        df.loc[df['label_name'] == label_name, 'label'] = label


    dfs[img_name] = df

  return dfs



def augment(img):

    ###############################################
    # define an augmentation pipeline
    aug_pipeline = iaa.Sequential([
        iaa.Sometimes(1.0, iaa.GaussianBlur((0.5, 0.8))), # apply Gaussian blur with a sigma between 0 and 1 to 50% of the images
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(2, 5))),
        iaa.Sometimes(0.7, iaa.AddToHueAndSaturation((-5, 5))),
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 0.5), lightness=(0.25, 1.0))), # sharpen images
    ],
    random_order=True # apply the augmentations in random order
    )

    # extra augs
    bright_contrast = A.RandomBrightnessContrast(p = 0.3, brightness_limit = 0.08, contrast_limit = 0.2) # random brightness and contrast
    gamma = A.RandomGamma(p = 0.3, gamma_limit = [30, 150]) # random gamma
    clahe = A.CLAHE(p = 0.2, clip_limit=6) # CLAHE (see https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)
    # blur = A.Blur(p = 0.2)

    # apply augmentation pipeline to sample image
    img_aug = aug_pipeline.augment_image(img)

    # apply extra augs
    img_aug = bright_contrast(image = img_aug)['image']
    img_aug = gamma(image = img_aug)['image']
    img_aug = clahe(image = img_aug)['image']
    # img_aug = blur(image = img_aug)['image']

    return img_aug




def rects_overlap(R1, R2):
  if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
    return False
  else:
    return True




def is_point_in_rect(polygon, p) :
   if (p[0] > polygon[0] and p[0] < polygon[2] and p[1] > polygon[1] and p[1] < polygon[3]) :
      return True
   else :
      return False

def get_rects_intersection(rect1, rect2):
    rect1, rect2 = Polygon(rect1), Polygon(rect2)
    intersection = rect1.intersection(rect2)
    if intersection.area <= 0:
      return False
    xx, yy = intersection.exterior.coords.xy
    x1 = np.min(xx)
    x2 = np.max(xx)
    y1 = np.min(yy)
    y2 = np.max(yy)
    return [x1, y1, x2, y2]
    




def generate_random_linear(low, high, prefer_high=False):
  weights = [1/(idx+1) for idx in range(high-low)]
  sw = sum(weights)
  weights = [w/sw for w in weights] # weights need to sum to 1
  if prefer_high:
    weights.reverse()
  return np.random.choice(range(low, high), 1, p=weights)[0]


def generate_samples(image, labels, num_samples=100, min_sample_dim=640, min_aspect_ratio=0.5, cell_dim=640, min_dim_only=False, prefer_big_samples=False, filter_small=False, filter_small_threshold=0.0, augment_img=False, augment_img_percentage=0.8):
  samples_bbs = []
  samples_classes = []
  samples = []
  max_bg_sample_ratio = 0.15
  # num samples adaptive to how crowd it is
  num_samples += len(labels)//10
  max_bg_samples = int(max_bg_sample_ratio*num_samples) 
  bg_sample_count = 0
  skipped = 0
  max_skipped = 5*num_samples
  sample_idx = 0
  while sample_idx < num_samples:
    # img shape is (height, width, 3)
    # generate top from 0 ~ image_height-min_sample_dim
    if image.shape[0] < min_sample_dim:
      top = 0
    else:
      top_low = 0
      top_high = image.shape[0]-min_sample_dim
      if not prefer_big_samples:
        top = np.random.randint(low=top_low, high=top_high)
      else:
        top = generate_random_linear(low=top_low, high=top_high)

    # generate left from 0 ~ image_width-min_sample_dim
    if image.shape[1] < min_sample_dim:
      left = 0
    else:
      left_low = 0
      left_high = image.shape[1]-min_sample_dim
      if not prefer_big_samples:
        left = np.random.randint(low=left_low, high=left_high)
      else:
        left = generate_random_linear(low=left_low, high=left_high)


    # generate sample dimensions (width and height) & preserve aspect ratio
    if not min_dim_only:
      w_low = min_sample_dim
      w_high = min(image.shape[1]-left, int((image.shape[0]-top)/min_aspect_ratio))
      if not prefer_big_samples:
        dim_w = np.random.randint(low=w_low, high=w_high)
      else:
        dim_w = generate_random_linear(low=w_low, high=w_high, prefer_high=True)

      h_low =max(min_sample_dim, int(dim_w*min_aspect_ratio))
      h_high = min(image.shape[0]-top, int(dim_w/min_aspect_ratio))
      if not prefer_big_samples:
        dim_h = np.random.randint(low=h_low, high=h_high)
      else:
        dim_h = generate_random_linear(low=h_low, high=h_high, prefer_high=True)  
        
    else:
      dim_w = min_sample_dim
      dim_h = min_sample_dim

    cell = image[top:top+dim_h, left:left+dim_w]
    cell = cv2.resize(cell, (cell_dim, cell_dim), interpolation = cv2.INTER_AREA)
    # augment
    if augment_img and random.random()<augment_img_percentage:
      cell = augment(cell)

    samples.append(cell)

    # filter labels outside sample
    height, width = image.shape[:-1]

    sample_coords = [left, top, left+dim_w, top+dim_h]

    x1, y1, x2, y2 = sample_coords
    sample_polygon = list(zip([x1,x2,x2,x1], [y1,y1,y2,y2]))


    bbs = []
    classes = []
    for i, row in labels.iterrows():

        x1, y1, x2, y2 = int(row['x1']*width), int(row['y1']*height), int(row['x2']*width), int(row['y2']*height)
        # make a list of tuples
        polygon = list(zip([x1,x2,x2,x1], [y1,y1,y2,y2]))


        # get part of polygon intersecting with the sample
        polygon = get_rects_intersection(polygon, sample_polygon)
        if not polygon:
          continue

        # get the coords of the intersection
        x1, y1, x2, y2 = polygon
        # rescale as ratios to sample size (xc, yc, w, h)
        bb = [
                (((x1+x2)/2) - left) / dim_w, 
                (((y1+y2)/2) - top) / dim_h, 
                (x2 - x1) / dim_w, 
                (y2 - y1) / dim_h
            ]
        bbs.append(bb)
        classes.append(row['label'])


    if len(bbs)>0 or bg_sample_count<max_bg_samples or skipped>max_skipped:
      samples_bbs.append(bbs)
      samples_classes.append(classes)
    else:
      skipped += 1
      sample_idx -= 1
    
    sample_idx += 1

  
  return samples, samples_bbs, samples_classes
    



def remove_extra_empty_files(image_folder, label_folder, max_empty_ratio=0.2):
    empty_files = []
    files = os.listdir(label_folder)
    print('Remove extra empty files')
    for f in tqdm(files): 
        file_path = os.path.join(label_folder, f) 
        with open(file_path, 'r') as file_obj:
          lines = file_obj.readlines()
          if len(lines) == 0:
              empty_files.append(f)

    random.shuffle(empty_files)
    for del_file in empty_files[0:min(len(empty_files), int(max_empty_ratio*len(files)))]:
        os.remove(os.path.join(image_folder, del_file.split('.')[0]+'.jpeg'))
        os.remove(os.path.join(label_folder, del_file))



def generate_data(anns, source_image_names, image_source_dir, image_destination_dir, label_destination_dir, num_samples=100, min_sample_dim=640, min_aspect_ratio=0.5, cell_dim=640, min_dim_only=False, prefer_big_samples=False, filter_small=False, filter_small_threshold=0.0, max_empty_ratio=0.2, augment_img=False, augment_img_percentage=0.8):
  for image_name in tqdm(source_image_names):
    source_image_path = os.path.join(image_source_dir, image_name)
    if not source_image_path.endswith(('.jpg', '.png', 'jpeg')):
      continue
    image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels = anns[image_name]
    samples, samples_bbs, samples_classes = generate_samples(image, labels, num_samples=num_samples, min_sample_dim=min_sample_dim, min_aspect_ratio=min_aspect_ratio, cell_dim=cell_dim, min_dim_only=min_dim_only, prefer_big_samples=prefer_big_samples, filter_small=filter_small, filter_small_threshold=filter_small_threshold, augment_img=augment_img, augment_img_percentage=augment_img_percentage)

    for i in range(len(samples)): 
      # save the sample
      sample = samples[i]
      sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
      sample_path = os.path.join(image_destination_dir, image_name.split('.')[0]+'_'+str(i)+'.'+image_name.split('.')[1])
      cv2.imwrite(sample_path, sample)

      # save the labels
      sample_bbs = samples_bbs[i]
      sample_classes = samples_classes[i]
      labels_path = os.path.join(label_destination_dir, image_name.split('.')[0]+'_'+str(i)+'.txt')
      with open(labels_path, 'w') as label_file:
        for j, bb in enumerate(sample_bbs):
          cls = str(int(sample_classes[j]))
          bb_line = ' '.join([cls] + [str(i) for i in bb]) + '\n'
          label_file.write(bb_line)
  
  remove_extra_empty_files(image_destination_dir, label_destination_dir, max_empty_ratio=max_empty_ratio)




