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
    draw.rectangle(((x1, y1), (x2, y2)), fill=color)
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
    draw.rectangle(shape, outline =color_code[row['label_name']], width=2)
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
  # get image widths and heights
  img_h = {p['file_name']:p['height'] for p in data['images']}
  img_w = {p['file_name']:p['width'] for p in data['images']}

  # initialize empty lists for each image
  bboxes = {img_ids[p['image_id']]:[] for p in data['annotations']}
  classes = {img_ids[p['image_id']]:[] for p in data['annotations']}

  # parse annotations and add them to corresponding image list in the dictionaries
  for p in data['annotations']:
    bboxes[img_ids[p['image_id']]].append(p['bbox'])
    classes[img_ids[p['image_id']]].append(p['category_id'])
  
  # if visualization will be done, create random but visually-separable color code and save it
  if visualize:
    print('Create New Color Code for Labels')
    color_code = getDistinctColors(list(cat_to_id.keys()))
    print(color_code)
    with open("color_code.json", "w") as outfile:
      json.dump(color_code, outfile)

  # create dataframe per image
  dfs = dict()

  if os.path.exists(vis_folder):
    shutil.rmtree(vis_folder)
  os.mkdir(vis_folder)

  for img_name in tqdm(bboxes.keys()):
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
        iaa.Sometimes(0.5, iaa.GaussianBlur((0, 0.5))), # apply Gaussian blur with a sigma between 0 and 1 to 50% of the images
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(2, 5))),
        iaa.Sometimes(0.7, iaa.AddToHueAndSaturation((-5, 5))),
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 0.5), lightness=(0.25, 1.0))), # sharpen images
    ],
    random_order=True # apply the augmentations in random order
    )

    # extra augs
    bright_contrast = A.RandomBrightnessContrast(p = 0.3, brightness_limit = 0.1, contrast_limit = 0.3) # random brightness and contrast
    gamma = A.RandomGamma(p = 0.3, gamma_limit = [30, 150]) # random gamma
    clahe = A.CLAHE(p = 0.2, clip_limit=6) # CLAHE (see https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)
    blur = A.Blur(p = 0.2)

    # apply augmentation pipeline to sample image
    img_aug = aug_pipeline.augment_image(img)

    # apply extra augs
    img_aug = bright_contrast(image = img_aug)['image']
    img_aug = gamma(image = img_aug)['image']
    img_aug = clahe(image = img_aug)['image']
    img_aug = blur(image = img_aug)['image']

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




def generate_samples(image, labels, num_samples=100, min_sample_dim=640, min_aspect_ratio=0.5, cell_dim=640, min_dim_only=False, filter_small=False, filter_small_threshold=0.0):
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
      top = np.random.randint(low=0, high=image.shape[0]-min_sample_dim)
    # generate left from 0 ~ image_width-min_sample_dim
    if image.shape[1] < min_sample_dim:
      left = 0
    else:
      left = np.random.randint(low=0, high=image.shape[1]-min_sample_dim)

    # generate sample dimensions (width and height) & preserve aspect ratio
    if not min_dim_only:
      dim_w = np.random.randint(low=min_sample_dim, high=min(image.shape[1]-left, (image.shape[0]-top)//min_aspect_ratio))
      dim_h = np.random.randint(low=max(min_sample_dim, int(dim_w*min_aspect_ratio)), high=min(image.shape[0]-top, int(dim_w/min_aspect_ratio)))
    else:
      dim_w = min_sample_dim
      dim_h = min_sample_dim

    cell = image[top:top+dim_h, left:left+dim_w]
    # cell_dim = min(dim_h, dim_w)
    cell = cv2.resize(cell, (cell_dim, cell_dim), interpolation = cv2.INTER_AREA)
    # augment
    # cell = augment(cell)

    samples.append(cell)

    # filter labels outside sample
    height, width = image.shape[:-1]

    sample_coords = [left, top, left+dim_w, top+dim_h]

    bbs = []
    classes = []
    for i, row in labels.iterrows():

        x1, y1, x2, y2 = int(row['x1']*width), int(row['y1']*height), int(row['x2']*width), int(row['y2']*height)
        
        # two classes only 0->aphid, 1->other(excluded,winged)
        if row['label'] == 1:
          cls = 0
        else:
          cls = 1

        # make a list of tuples
        polygon = list(zip([x1,x2,x2,x1], [y1,y1,y2,y2]))

        # rescale to image size
        polygon = [[int(point[0]), int(point[1])] for point in polygon]

        # get part of polygon intersecting with the sample
        polygon = [point for point in polygon if is_point_in_rect(sample_coords, point)]


        # check if polygon intersects or inside the sample
        if len(polygon)==4:
          polygon = np.array(polygon)
          # rescale as ratios to sample size
          bb = [
                  (((x1+x2)/2) - left) / dim_w, 
                  (((y1+y2)/2) - top) / dim_h, 
                  (x2 - x1) / dim_w, 
                  (y2 - y1) / dim_h
              ]
          
          bbs.append(bb)
          classes.append(cls)

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



def generate_data(anns, source_image_names, image_source_dir, image_destination_dir, label_destination_dir, num_samples=100, min_sample_dim=640, min_aspect_ratio=0.5, cell_dim=640, min_dim_only=False, filter_small=False, filter_small_threshold=0.0, max_empty_ratio=0.2):
  for image_name in tqdm(source_image_names):
    source_image_path = os.path.join(image_source_dir, image_name)
    image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels = anns[image_name]
    samples, samples_bbs, samples_classes = generate_samples(image, labels, num_samples=num_samples, min_sample_dim=min_sample_dim, min_aspect_ratio=min_aspect_ratio, cell_dim=cell_dim, min_dim_only=min_dim_only, filter_small=filter_small, filter_small_threshold=filter_small_threshold)

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




