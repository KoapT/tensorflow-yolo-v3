# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image
import cv2


def continue_time(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('\'{}\' time consumption:{}'.format(func.__name__, end_time - start_time))
        return result

    return wrapper


def get_boxes_and_inputs_pb(frozen_graph):
    with frozen_graph.as_default():
        boxes = tf.get_default_graph().get_tensor_by_name("output_boxes:0")
        inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")

    return boxes, inputs


def get_boxes_and_inputs(model, num_classes, size, data_format):
    inputs = tf.placeholder(tf.float32, [1, size, size, 3])

    with tf.variable_scope('detector'):
        detections = model(inputs, num_classes,
                           data_format=data_format)

    boxes = detections_boxes(detections)

    return boxes, inputs

@continue_time
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def freeze_graph(sess, output_graph):
    output_node_names = [
        "output_boxes"
    ]

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names
    )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("{} ops written to {}.".format(len(output_graph_def.node), output_graph))


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)  # np.ndarray

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for vari in batch_norm_vars:
                    shape = vari.shape.as_list()
                    num_params = np.prod(shape)
                    vari_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(
                        tf.assign(vari, vari_weights, validate_shape=True))  # tf.sssign() 给变量赋值

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(
                    tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def detections_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    center_x, center_y, width, height, attrs = tf.split(
        detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1, name="output_boxes")
    return detections


def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou


# b1 = (1450.0, 848.0, 1483.0, 874.0)
# b2 = (1695.4978030000002, 815.072266,1717.4360350000002,  842.191162)
# print(_iou(b1,b2))

#@continue_time
def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4) -> dict:
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array[batch,boxes,(4+1+2)],
    first 4 values in 3rd dimension are bbox attrs, 5th is confidence, 6/7th classifications
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    results = []
    for i, image_pred in enumerate(predictions):
        result = {}
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-
        1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]  # 得到该类的所有boxes
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]  # np.argsort() 对数列从小到大排序，返回序号
            cls_scores = cls_boxes[:, -1]  # 最后一列表示得分（置信度）
            cls_boxes = cls_boxes[:, :-1]  # 前四列是box位置信息

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]  # 先选择置信度最大的box和score作为基准
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
        results.append(result)
    # print (results)
    return results


def load_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

#@continue_time
def draw_boxes(boxes, img, cls_names, detection_size, keep_aspect_ratio):
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for cls, bboxs in boxes.items():
        color = colors[cls % 6]
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.size),
                                           keep_aspect_ratio)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(
                cls_names[cls], score * 100), fill=color)

#@continue_time
def draw_boxes_cv2(boxes: dict, img: np.ndarray, cls_names: dict, detection_size: tuple, keep_aspect_ratio=False):
    # draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for cls, bboxs in boxes.items():
        color = colors[cls % 6]
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.shape[:2][::-1]),  # (h,w)->(w,h)
                                           keep_aspect_ratio)
            box = [max(1, box[0]), max(1, box[1]),
                   min(img.shape[1] - 1, box[2]), min(img.shape[0] - 1, box[3])]
            left_top, right_bottom = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(img, left_top, right_bottom, color, 2)
            cv2.putText(img, '{}{:.2f}%'.format(cls_names[cls].strip(), score * 100),
                        left_top, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            print('name:{0},\t location:{1[0]:>4d},{1[1]:>4d},{1[2]:>4d},{1[3]:>4d},\t confidence:{2:.2%}'
                  .format(cls_names[cls].strip(), box, score))


def convert_to_original_size(box: np.ndarray, size: np.ndarray, original_size: np.ndarray, keep_aspect_ratio) -> list:
    if keep_aspect_ratio:
        box = box.reshape(2, 2)
        box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
        box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)
    else:
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
    return [int(i) for i in box.reshape(-1)]


def letter_box_image(image: Image.Image, output_height: int, output_width: int, fill_value) -> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height) / image.size[1]
    width_ratio = float(output_width) / image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top:pad_top + fit_height, pad_left:pad_left + fit_width] = fit_image
    return to_return

#@continue_time
def resize_cv2(image: np.array, output_size: tuple, keep_aspect_ratio=False, fill_value=128) -> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """
    output_width, output_height = output_size[0], output_size[1]
    if keep_aspect_ratio:
        height_ratio = float(output_height) / image.shape[0]
        width_ratio = float(output_width) / image.shape[1]
        fit_ratio = min(width_ratio, height_ratio)
        fit_height = int(image.shape[0] * fit_ratio)
        fit_width = int(image.shape[1] * fit_ratio)
        fit_image = cv2.resize(image, (fit_width, fit_height))

        if isinstance(fill_value, int):
            fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

        to_return = np.tile(fill_value, (output_height, output_width, 1))
        pad_top = int(0.5 * (output_height - fit_height))
        pad_left = int(0.5 * (output_width - fit_width))
        to_return[pad_top:pad_top + fit_height, pad_left:pad_left + fit_width] = fit_image
        return to_return
    else:
        return cv2.resize(image, (output_width, output_height))


def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size) -> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0] / ori_image_size[0], current_size[1] / ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos
