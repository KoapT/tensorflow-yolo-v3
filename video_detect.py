# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import cv2
import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'video_path', '/home/tk/Share/record/a/video.dav', 'Input image')
tf.app.flags.DEFINE_string(
    'class_names', 'bird.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'frozen_model', 'saved_model_pb/bird_tiny_3l.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_integer(
    'size', 608, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.3, 'Confidence threshold')
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
    cap = cv2.VideoCapture(FLAGS.video_path)
    classes = utils.load_names(FLAGS.class_names)
    frozenGraph = utils.load_graph(FLAGS.frozen_model)
    boxes, inputs = utils.get_boxes_and_inputs_pb(frozenGraph)

    with tf.Session(graph=frozenGraph, config=config) as sess:
        while True:
            ret, frame = cap.read()
            if ret:
                t1 = time.time()
                frame1 = frame[:, :, ::-1]  # from BGR to RGB
                # frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print('\'BGR2RGB\' time consumption:', time.time() - t1)
                img_resized = utils.resize_cv2(frame1, (FLAGS.size, FLAGS.size),
                                               keep_aspect_ratio=FLAGS.keep_aspect_ratio)
                img_resized = img_resized[np.newaxis, :]
                t0 = time.time()
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: img_resized})  # get the boxes whose confidence > 0.005
                filtered_boxes = utils.non_max_suppression(detected_boxes,
                                                           confidence_threshold=FLAGS.conf_threshold,
                                                           iou_threshold=FLAGS.iou_threshold)[0]  # boxes' filter by NMS
                print('\'detection\' time consumption:', time.time() - t0)
                utils.draw_boxes_cv2(filtered_boxes, frame, classes, (FLAGS.size, FLAGS.size), FLAGS.keep_aspect_ratio)
                print('\n\n\n')
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
