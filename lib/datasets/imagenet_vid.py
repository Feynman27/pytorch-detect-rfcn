import sys
import os
import argparse
import shutil
import h5py
import numpy as np
import pandas as pd
import scipy.misc as sp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob
import pdb

args=None
'''
Base class for Parsing all the datasets
'''

class DatasetParser:
    def __init__(self, data_dir, _data_splits=[0.7, 0.1, 0.2]):

        self._classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']
        self._classes_map = ['__background__',  # always index 0
                            'n02691156', 'n02419796', 'n02131653', 'n02834778',
                            'n01503061', 'n02924116', 'n02958343', 'n02402425',
                            'n02084071', 'n02121808', 'n02503517', 'n02118333',
                            'n02510455', 'n02342885', 'n02374451', 'n02129165',
                            'n01674464', 'n02484322', 'n03790512', 'n02324045',
                            'n02509815', 'n02411705', 'n01726692', 'n02355227',
                            'n02129604', 'n04468005', 'n01662784', 'n04530566',
                            'n02062744', 'n02391049']

        self._name_to_class = dict(zip(self._classes_map, self._classes))
        # Class name to index
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        # Structure to hold class statistics
        self._class_counter = dict()
        # Data location
        self.data_dir = data_dir
        # Initialize all the sub-directory structure following PASCAL VOC
        self.init_paths()
        # Data splits in the form of [train, val, test]
        self.data_splits = _data_splits
        assert sum(self.data_splits) == 1.0, "All the splits must sum to 1.0"

        # Rest of this data must be filled by class specific to each dataset
        self.dataset_name = None  # Prepend images files with this name
        self.im_list = None
        # To store information for all images, each image is a dictionary key
        # which stores a dictionary for each image containing class and
        # bounding box information
        self.im_data = dict()
        # Directory of image data
        self.im_dir = None

    def init_paths(self):
        '''
        Initialize all the paths under data_dir directory to replicate most of the Pascal VOC structure
        :return:
        '''
        # Assert that a valid data_dir is passed
        assert os.path.isdir(self.data_dir), "A valid directory required. No directory named {}".format(self.data_dir)
        return

class ImagenetDET(DatasetParser):
    def __init__(self, data_dir, dataset_path):
        # Calling the base class constructor first
        DatasetParser.__init__(self, data_dir)
        # Store the dataset path
        self.dataset_path = dataset_path
        self.unique_classes = []

        # Get all the images that are present in the dataset
        self.im_list = []
        self.img_to_annot_map = {}
        self.det_list = {'train':{}, 'val':{}}
        self.get_det_list()

    def write_to_file(self):
        for segment in self.det_list:
            fname = os.path.join(args.output_path,'DET',segment) + "_det.txt"
            if os.path.exists(fname):
                os.remove(fname)
            print("Writing to {}".format(fname))
            counter = 0
            for img_set in self.det_list[segment]:
                for img_name in self.det_list[segment][img_set]:
                    with open(fname,"a+") as f:
                        counter+=1
                        if segment=='val':
                            f.write(img_name.split('.')[0] + " " + str(counter) + "\n")
                        else:
                            f.write(os.path.join(img_set,img_name.split('.')[0]) \
                                + " " + str(counter) + "\n")
        pdb.set_trace()

    def merge_train_val(self):
        raise NotImplementedError


    def get_det_list(self):
        '''
        Get images in Imagenet DET.
        '''
        np.random.seed(1)
        # Iterate over train/val/test
        self.det_data_path = os.path.join(self.dataset_path, 'DET') 
        for segment in os.listdir(self.det_data_path):
            if segment not in self.det_list: continue
            # Build list of images for each segment
            seg_path = os.path.join(self.det_data_path, segment)
            n_images = 0
            for i,img in enumerate(os.walk(seg_path)):
                if (i==0 and segment=='train') or len(img[2])==0:
                    continue
                print("Found {} images in directory {}".format(len(img[2]), img[0]))
                img_list = sorted(img[2])

                n_images += len(img_list)
                if segment=='train':
                    set_name=img[0][img[0].index('train')+len('train/'):]
                elif segment=='val':
                    #set_name=img[0][img[0].index('val')+len('val/'):]
                    set_name = os.path.basename(img[0])
                else:
                    raise NotImplementedError
                if set_name not in self.det_list[segment]:
                    self.det_list[segment][set_name]=[]
                self.det_list[segment][set_name] = img_list
                #self.det_list[segment].update(img_list)


            print("Total images in DET {}:{}".format(segment,n_images))



class ImagenetVID(DatasetParser):
    def __init__(self, data_dir, dataset_path):
        # Calling the base class constructor first
        DatasetParser.__init__(self, data_dir)
        # Store the dataset path
        self.dataset_path = dataset_path
        self.unique_classes = []

        # Get all the images that are present in the dataset
        self.im_list = []
        self.img_to_annot_map = {}
        self.vid_list = {'train':{}, 'val':{}, 'test':{}}
        self.get_vid_list()
        if args.use_det:
            self.det_image_list=[]
            self.get_intersection_with_det_list()

    def write_to_file(self):
        for segment in self.vid_list:
            fname = os.path.join(args.output_path,'VID',segment) + ".txt"
            if os.path.exists(fname):
                os.remove(fname)
            print("Writing to {}".format(fname))
            for video in self.vid_list[segment]:
                if len(self.vid_list[segment][video])==0: continue
                last_frame=int(self.vid_list[segment][video][-1].split('.')[0])
                for frame in self.vid_list[segment][video]:
                    frame_number = int(frame.split('.')[0])
                    with open(fname,"a+") as f:
                        f.write(os.path.join(video,frame.split('.')[0]) \
                                + " 1" + " " + str(frame_number) + " " + str(last_frame) + "\n")
        if args.use_det:
            segment='train' # only use train samples from DET
            fname = os.path.join(args.output_path,'DET',segment) + ".txt"
            if os.path.exists(fname):
                os.remove(fname)
            print("Writing to {}".format(fname))
            for train_img in self.det_image_list:
                with open(fname,"a+") as f:
                    f.write(train_img)



    def merge_train_val(self):
        raise NotImplementedError

    def get_intersection_with_det_list(self):
        '''
        Get images in Imagenet DET that overlap Imagenet VID classes.
        '''
        np.random.seed(1)

        det_classes_in_vid = set([2,4,20,24,
				  26,33,37,39,
                                  58,59,64,70,
                                  74,84,92,103,
                                  105,113,114,141,
                                  144,155,159,166, 
                                  182,185,188,197,
                                  198,200])
        assert len(det_classes_in_vid)==len(self._classes_map)-1, 'Length of overlap set in DET!=VID.'
        # Iterate over train/val/test
        imageset_path = self.dataset_path.replace('Data', 'ImageSets')
        self.det_data_path = os.path.join(imageset_path, 'DET') 
        # iterate over train lists for each category in DET
        for class_file in os.listdir(self.det_data_path):
            basename = class_file.split('.')[0].split('_')
            if len(basename)<2:
                continue
            try:
                class_index = int(basename[1])
            except:
                print('Warning: {} does not represent an integer. Skipping.'.format(basename[1]))
                continue
            # use only classes in DET that are also in VID
            if class_index not in det_classes_in_vid:
                continue

            with open(os.path.join(self.det_data_path, class_file)) as f: 
                class_image_list = f.readlines()
            class_image_list = [x for x in class_image_list \
                                    if '-1' not in x.split(' ')[-1]] 
            print('Found {} images for class {}'.format(len(class_image_list), class_index))
            # sample at most max_det_samples samples per class
            if len(class_image_list)>=args.max_det_samples:
                print('Subsampling class examples in DET')
                class_image_list = np.random.choice(class_image_list, size=args.max_det_samples, 
                        replace=False)
                class_image_list = class_image_list.tolist()

            # keep image list over all classes
            self.det_image_list += class_image_list

        print("Total images in DET train after sampling: {}".format(len(self.det_image_list)))

    def get_vid_list(self):
        '''
        Get frames from snippets in Imagenet VID, optionally subsampled.
        '''
        np.random.seed(1)
        # Iterate over train/val/test
        self.vid_data_path = os.path.join(self.dataset_path, 'VID') 
        for segment in os.listdir(self.vid_data_path):
            if segment not in self.vid_list: continue
            # Build list of video snippets for each segment
            seg_path = os.path.join(self.vid_data_path, segment)
            n_frames = 0
            for i,vid in enumerate(os.walk(seg_path)):
                if i==0 or len(vid[2])==0:
                    print(vid[0])
                    continue
                # sort from 1st frame --> last frame in video snippet
                frame_list = sorted(vid[2])
                #if os.path.basename(vid[0]) not in self.vid_list[segment]:
                #    self.vid_list[segment][os.path.basename(vid[0])]=[]
                #self.vid_list[segment][os.path.basename(vid[0])]=sorted(vid[2])
                if args.frames_per_video != -1:
                    frame_list = frame_list[0::int(np.ceil(len(frame_list) / float(args.frames_per_video)))]
                    # consecutive frame
                    # Sample starting frame
                    #if len(frame_list)>args.frame_per_video:
                    #    start_frame = np.random.choice(len(frame_list)-args.frames_per_video, size=1)[0]
                    #    frame_list = frame_list[start_frame:start_frame+int(args.frames_per_video)]
                    #else:
                    #    start_frame=0
                    #    frame_list = frame_list[start_frame:]

                n_frames += len(frame_list)
                if segment=='train':
                    snippet_name=vid[0][vid[0].index('train')+len('train/'):]
                elif segment=='val' or segment=='test':
                    #set_name=img[0][img[0].index('val')+len('val/'):]
                    snippet_name = os.path.basename(vid[0])
                else:
                    raise NotImplementedError
                if snippet_name not in self.vid_list[segment]:
                    self.vid_list[segment][snippet_name]=[]
                self.vid_list[segment][snippet_name] = frame_list


            print("Total frames in VID {}:{}".format(segment,n_frames))


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Build ImageNet VID dataset.')
    parser.add_argument('--dataset', dest='dataset',
                        help='Name of the dataset',
                        default='ImagenetVID', choices=['ImagenetVID', 
                            'ImagenetVID+ImagenetDET'],type=str)
    parser.add_argument('--input_path', dest='input_path',
                        help='Path to input data.',
                        default='./data/ILSVRC/Data/',
                        type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='Path to where the new data will be stored.',
                        default='./data/ILSVRC/ImageSets',
                        type=str)
    parser.add_argument('--frames_per_video', dest='frames_per_video',
                        help='Number of frames to use per video. Default all.',
                        default=-1, type=int)
    parser.add_argument('--max_det_samples', dest='max_det_samples',
                        help='Maximum number of samples per class to sample from Imagenet DET',
                        default=2000, type=int)
    parser.add_argument('--use_det', dest='use_det',
                        help='Use images from DET.',
                        default=True, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# To get the name of class from string
def str_to_classes(str):
    curr_class = None
    try:
        curr_class = getattr(sys.modules[__name__], str)
    except:
        print "Dataset class is not implemented"
    return curr_class

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    data_path = args.output_path
    datasets = args.dataset.split('+')
    dataset_paths = [args.input_path]
    # Process all the datasets
    for dataset in datasets:
        dataset_path = dataset_paths[0]
        curr_dataset = str_to_classes(dataset)(data_path, dataset_path)
        curr_dataset.write_to_file()
