import numpy as np
from matplotlib import pyplot as plt

from bounding_box_utils.bounding_box_utils import iou

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_patch_sampling_ops import *
from data_generator.object_detection_2d_geometric_ops import *
from data_generator.object_detection_2d_photometric_ops import *
from data_generator.object_detection_2d_image_boxes_validation_utils import *
from data_generator.data_augmentation_chain_original_ssd import *

dataset = DataGenerator(labels_output_format = ('class_id', 'xmin', 'ymin', 'xmax', 'ymax'))

images_dir         = 'E:/Hamed/Projects/Python/License Plate Detection/License-Plate-Detection/Final Plates/background.csv'
annotations_dir    = 'E:/Hamed/Projects/Python/License Plate Detection/License-Plate-Detection/Final Plates/summary.csv'
image_set_filename = 'E:/Hamed/Projects/Python/License Plate Detection/License-Plate-Detection/Final Plates/imageset.txt'

characterList = np.array('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split(' '))
numbersList = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype = int)
func = np.vectorize(str)
classes = np.concatenate([np.array(['background']), func(numbersList), characterList])

dataset.parse_csv(images_dir = images_dir,
                  labels_filename = image_set_filename,
                  input_format = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                  include_classes = classes[0],
                  ret = False)