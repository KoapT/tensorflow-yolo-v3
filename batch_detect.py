# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import os
import cv2
from tqdm import tqdm
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_imgpath', 'data/videoframe', 'Input image')
tf.app.flags.DEFINE_string(
    'output_imgpath', 'data/videoframe_out', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'bird.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'frozen_model', 'saved_model_pb/bird.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_integer(
    'size', 608, 'Image size')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'Batch size')
tf.app.flags.DEFINE_float(
    'conf_threshold', 0.3, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.3, 'IoU threshold')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0, 'Gpu memory fraction to use')
tf.app.flags.DEFINE_bool(
    'keep_aspect_ratio', False, 'To keep the w&h ratio while resizing')

try:
    os.mkdir(FLAGS.output_imgpath)
except:
    pass


def main(argv=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        # inter_op_parallelism_threads=0,
        # intra_op_parallelism_threads=0,
        # device_count={"CPU": 6}
    )

    classes = utils.load_names(FLAGS.class_names)
    input_size = (FLAGS.size, FLAGS.size)
    img_pathes = [path for path in os.listdir(FLAGS.input_imgpath)
                  if path.endswith(('.jpg', '.png', '.bmp'))]
    num_imgs = len(img_pathes)
    batch_size = FLAGS.batch_size
    img_list = []

    img_batch_all = np.zeros((num_imgs, FLAGS.size, FLAGS.size, 3))
    for k in range(num_imgs):
        img_array = cv2.imread(os.path.join(FLAGS.input_imgpath, img_pathes[k]))
        img_list.append(img_array)
        img_batch_all[k] = utils.resize_cv2(img_array, input_size)[:, :, ::-1]

    frozenGraph = utils.load_graph(FLAGS.frozen_model)
    boxes, inputs = utils.get_boxes_and_inputs_pb(frozenGraph)

    with tf.Session(graph=frozenGraph, config=config) as sess:
        for i in range(0, num_imgs, batch_size):
            if i < num_imgs - batch_size:
                img_batch = img_batch_all[i:i + batch_size]
            else:
                img_batch = img_batch_all[i:]

            detected_boxes = sess.run(boxes, feed_dict={inputs: img_batch})
            filtered_boxes = utils.non_max_suppression(detected_boxes,
                                                       confidence_threshold=FLAGS.conf_threshold,
                                                       iou_threshold=FLAGS.iou_threshold)
            for n, bboxes in enumerate(filtered_boxes):
                img = img_list[i + n]
                img_name = img_pathes[i + n]
                utils.draw_boxes_cv2(bboxes, img, classes, input_size, keep_aspect_ratio=FLAGS.keep_aspect_ratio)
                # cv2.imshow('image_{}'.format(img_name), img)
                cv2.imwrite(os.path.join(FLAGS.output_imgpath, 'out_' + img_name), img)
                print('{} has been processed !'.format(img_name))
                print('#'*20)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
