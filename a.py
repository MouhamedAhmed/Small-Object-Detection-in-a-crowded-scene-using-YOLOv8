import os
for i in os.listdir('dataset/yolo_dataset_random_tiles/train/labels'):
    f = os.path.join('dataset/yolo_dataset_random_tiles/train/labels', i)
    file1 = open(f, 'r')
    Lines = file1.readlines()
    # Strips the newline character
    for line in Lines:
        if int(line.split()[0]) == 2:
            print(i)
