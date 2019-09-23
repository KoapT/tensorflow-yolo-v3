# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', 'data/bb.jpg', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', 'data/out/aaa.jpg', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'bird.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'frozen_model', 'saved_model_pb/bird.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_integer(
    'size', 608, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', .2, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.3, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0, 'Gpu memory fraction to use')
tf.app.flags.DEFINE_bool(
    'keep_aspect_ratio', False, 'To keep the w&h ratio while resizing')


def main(argv=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        # inter_op_parallelism_threads=0,
        # intra_op_parallelism_threads=0,
        # device_count={"CPU": 6}
    )

    img = Image.open(FLAGS.input_img)
    if FLAGS.keep_aspect_ratio:
        img_resized = utils.letter_box_image(img, FLAGS.size, FLAGS.size, 128)
        img_resized = img_resized.astype(np.float32)
    else:
        img_resized = img.resize((FLAGS.size, FLAGS.size), Image.BILINEAR)
        img_resized = np.asarray(img_resized, dtype=np.float32)

    classes = utils.load_names(FLAGS.class_names)
    frozenGraph = utils.load_graph(FLAGS.frozen_model)

    boxes, inputs = utils.get_boxes_and_inputs_pb(frozenGraph)

    with tf.Session(graph=frozenGraph, config=config) as sess:
        t0 = time.time()
        detected_boxes = sess.run(
            boxes, feed_dict={inputs: [img_resized]})

    print("Predictions found in {:.2f}s".format(time.time() - t0))

    filtered_boxes = utils.non_max_suppression(detected_boxes,
                                               confidence_threshold=FLAGS.conf_threshold,
                                               iou_threshold=FLAGS.iou_threshold)[0]

    utils.draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), FLAGS.keep_aspect_ratio)

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
