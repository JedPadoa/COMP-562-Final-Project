import cv2
import numpy as np
from gdx_training import gdx_nn
from gda_training import gda_nn
from gdm_training import gdm_nn
from gd_training import gd_nn

data_file = input("Enter path of letter.data file: ")

def processing(input):

    img_resize_factor = 12
    start, end = 6, -1
    height, width = 16, 8

    with open(input, 'r') as f:
        for line in f.readlines():
            data = np.array([255*float(x) for x in line.split('\t')[start:end]])
            img = np.reshape(data, (height, width))
            scaled = cv2.resize(img, None, fx = img_resize_factor, fy = img_resize_factor)
            print(line)
            cv2.imshow('Image', scaled)

            c = cv2.waitKey()
            if c == 27:
                break
				
processing(data_file)

num_data = 100
orig_labels = 'omandig'
num_orig_labels = len(orig_labels)

num_train = int(0.9*num_data)
num_test = num_data - num_train

start = 6
end = -1

data = []
labels = []

with open(data_file, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels:
            continue
        label = np.zeros((num_orig_labels, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)
        current = np.array([float(x) for x in list_vals[start:end]])
        data.append(current)
        if len(data) >= num_data:
            break

data = np.asfarray(data)
labels = np.array(labels).reshape(num_data, num_orig_labels)

num_dims = len(data[0])

gdx_nn(data, num_orig_labels, num_train, labels, num_test, orig_labels)
gda_nn(data, num_orig_labels, num_train, labels, num_test, orig_labels)
gdm_nn(data, num_orig_labels, num_train, labels, num_test, orig_labels)
gd_nn(data, num_orig_labels, num_train, labels, num_test, orig_labels)
