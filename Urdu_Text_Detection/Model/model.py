import numpy as np
import pandas as pd

import tensorflow as tf

import loader

def yolo_body(input_shape):
    X_input = tf.keras.Input(input_shape)
    X = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(X_input)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.MaxPool2D()(X)
    X = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.MaxPool2D()(X)
    X = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same')(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.MaxPool2D()(X)
    X = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same')(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Conv2D(filters=5, kernel_size=1, padding='same')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)
    return model

def yolo_head(yolo_outputs, img_shape):
    confidences = yolo_outputs[..., 0]
    confidences = tf.math.sigmoid(confidences)
    y = yolo_outputs[..., 1]
    x = yolo_outputs[..., 2]
    y += np.arange(yolo_outputs.shape[1]).reshape(1, -1, 1)
    x += np.arange(yolo_outputs.shape[2]).reshape(1, 1, -1)
    grid_size = img_shape[0] / yolo_outputs.shape[1]
    y *= grid_size
    x *= grid_size
    h = yolo_outputs[..., 3] * grid_size
    w = yolo_outputs[..., 4] * grid_size
    return confidences, tf.stack([y, x, h, w], axis=-1)

def yolo_boxes_to_corners(boxes):
    y, x, h, w = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    ymin = y - h / 2
    ymax = y + h / 2
    xmin = x - w / 2
    xmax = x + w / 2
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

def yolo_nms(scores, corners, boxes, max_boxes, score_threshold, iou_threshold):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(corners, scores, max_boxes_tensor, iou_threshold, score_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    return scores, boxes

def yolo_eval(yolo_output, img_shape, max_boxes = 10, score_threshold = 0.2, iou_threshold = 0.2):
    scores, boxes = yolo_head(tf.reshape(yolo_output, (1,) + yolo_output.shape), img_shape)
    # scores, boxes = yolo_output[..., 0], yolo_output[..., 1:]
    scores, boxes = tf.reshape(scores, (-1,)), tf.reshape(boxes, (-1, 4))
    corners = yolo_boxes_to_corners(boxes)
    scores, boxes = yolo_nms(scores, corners, boxes, max_boxes, score_threshold, iou_threshold)
    return scores, boxes

def yolo_cost(y, yhat, beta = 0.5):
    confidences_yhat = tf.math.sigmoid(yhat[..., 0])
    boxes_yhat = yhat[..., 1:]
    confidences_y = y[..., 0]
    boxes_y = y[..., 1:]

    confidences_cost = tf.reduce_mean(tf.keras.losses.binary_focal_crossentropy(confidences_y, confidences_yhat, apply_class_balancing=True))
    boxes_cost = tf.reduce_mean(confidences_y * tf.keras.losses.mean_squared_logarithmic_error(boxes_y, boxes_yhat))
    return beta * float(confidences_cost) + (1 - beta) * float(boxes_cost)

def save(model, link):
    model.save(rf'Urdu_Text_Detection\Model\{link}')

def load(link, yolo_cost = yolo_cost):
    custom_objects = {'yolo_cost': yolo_cost}
    return tf.keras.models.load_model(rf'Urdu_Text_Detection\Model\{link}', custom_objects=custom_objects)


if __name__ == '__main__':

    img_shape = (64, 96, 3)

    # model = yolo_body(img_shape)
    model = load('saved_model')

    train_x, train_y, orig_train_y, test_x, test_y, orig_test_y = loader.load_data()

    opt = tf.keras.optimizers.Adam(learning_rate=0.000005)
    model.compile(optimizer=opt, loss=yolo_cost)
    model.fit(train_x, train_y, batch_size=32, epochs=10)

    model.evaluate(train_x, train_y)
    model.evaluate(test_x, test_y)

    save(model, 'saved_model')

    # yhat = model.predict(train_x)
    # print(train_y[1, ..., 0])
    # print(tf.math.sigmoid(yhat[1, ..., 0]))
    # print(train_y[1, ..., 1])
    # print(yhat[1, ..., 1])
    # print(train_y[1, ..., 2])
    # print(yhat[1, ..., 2])
    # print(train_y[1, ..., 3])
    # print(yhat[1, ..., 3])

    # yhat = model.predict(train_x)
    # scores, boxes = yolo_eval(yhat[1], img_shape, score_threshold=0.2, iou_threshold=0.2)
    # print(scores)
    # print(boxes)