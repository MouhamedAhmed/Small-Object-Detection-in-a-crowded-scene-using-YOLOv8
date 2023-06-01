import argparse
import os
from preprocess_utils import *
import yaml


def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-config_path', '--config_path', type=str, help='yaml config path that contains preprocessing configs', default = 'config.yaml')
    args = argparser.parse_args()

    # read configs
    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)


    dataset_path = configs['dataset_path']
    img_path = os.path.join(dataset_path, configs['img_path'])
    json_path = os.path.join(dataset_path, configs['json_path'])

    # parse dataset json file and visualize groundtruth (if required)
    anns = parse_json(
                        json_path, 
                        img_path, 
                        labelmap=configs['labelmap'], 
                        visualize=configs['gt_vis'], 
                        vis_folder=os.path.join(dataset_path, configs['gt_vis_folder'])
                    )
    
    images_names = list(anns.keys())

    # split dataset
    images_train, images_valtest = train_test_split(images_names, test_size=0.1, random_state=42)
    images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)

    # create folders for the new dataset
    if configs['const_tiles']:
        folder = 'yolo_dataset_const_tiles'
    else:
        folder = 'yolo_dataset_random_tiles'
    
    if os.path.exists(os.path.join(dataset_path, folder)):
        shutil.rmtree(os.path.join(dataset_path, folder))
    os.mkdir(os.path.join(dataset_path, folder))

    train_destination_path = os.path.join(dataset_path, folder, 'train')
    valid_destination_path = os.path.join(dataset_path, folder, 'valid')
    test_destination_path = os.path.join(dataset_path, folder, 'test')
    os.mkdir(os.path.join(train_destination_path))
    os.mkdir(os.path.join(valid_destination_path))
    os.mkdir(os.path.join(test_destination_path))

    train_images_destination_path = os.path.join(dataset_path, folder, 'train', 'images')
    os.mkdir(os.path.join(train_images_destination_path))
    val_images_destination_path = os.path.join(dataset_path, folder, 'valid', 'images')
    os.mkdir(os.path.join(val_images_destination_path))
    test_images_destination_path = os.path.join(dataset_path, folder, 'test', 'images')
    os.mkdir(os.path.join(test_images_destination_path))

    train_labels_destination_path = os.path.join(dataset_path, folder, 'train', 'labels')
    os.mkdir(os.path.join(train_labels_destination_path))
    val_labels_destination_path = os.path.join(dataset_path, folder, 'valid', 'labels')
    os.mkdir(os.path.join(val_labels_destination_path))
    test_labels_destination_path = os.path.join(dataset_path, folder, 'test', 'labels')
    os.mkdir(os.path.join(test_labels_destination_path))

    # generate data
    print('Training Dataset Generation')
    generate_data(  
                    anns,
                    images_train, 
                    img_path, 
                    train_images_destination_path, 
                    train_labels_destination_path, 
                    num_samples=configs['num_samples'], 
                    min_sample_dim=configs['min_sample_dim'], 
                    cell_dim=configs['cell_dim'], 
                    min_dim_only=configs['const_tiles'], 
                    filter_small=configs['filter_small'], 
                    filter_small_threshold=configs['filter_small_threshold'], 
                    max_empty_ratio=configs['max_empty_ratio']
                )

    print('Validation Dataset Generation')
    generate_data(
                    anns,
                    images_val, 
                    img_path, 
                    val_images_destination_path, 
                    val_labels_destination_path, 
                    num_samples=configs['num_samples'], 
                    min_sample_dim=configs['min_sample_dim'], 
                    cell_dim=configs['cell_dim'], 
                    min_dim_only=configs['const_tiles'], 
                    filter_small=configs['filter_small'], 
                    filter_small_threshold=configs['filter_small_threshold'], 
                    max_empty_ratio=configs['max_empty_ratio']
                )

    print('Testing Dataset Generation')
    generate_data(
                    anns,
                    images_test, 
                    img_path, 
                    test_images_destination_path, 
                    test_labels_destination_path, 
                    num_samples=configs['num_samples'], 
                    min_sample_dim=configs['min_sample_dim'], 
                    cell_dim=configs['cell_dim'], 
                    min_dim_only=configs['const_tiles'], 
                    filter_small=configs['filter_small'], 
                    filter_small_threshold=configs['filter_small_threshold'], 
                    max_empty_ratio=configs['max_empty_ratio']
                )



if __name__ == '__main__':
    main()

# python3.10 generate_data.py --config_path 'config.yaml'