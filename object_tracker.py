""" this code is used to detect mask and nomask person and give you count in real time and at the end of
 . you will get total mask and no mask count.

 Referance of thi code : https://github.com/theAIGuysCode/yolov3_deepsort

 In this code yolov3 used to detect a two classe for "mask" and "nomask"and For tracking DeepSORTis used. """

import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

cv2.namedWindow('output', cv2.WINDOW_NORMAL)


def main(_argv):
    # Definition of the parameters
    cnt_mask, cnt_nomask = 0, 0
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        print("1")
        vid = cv2.VideoCapture(FLAGS.video)

    out = None
    mask_id, nomask_id = set(), set()
    mask_cnt_r, Nomask_cnt_r = 0, 0
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    count = 0
    while True:
        _, img = vid.read()


        if img is None:
            img_w, img_h = 1920,1080
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            label_t="Mask"
            label_t_n="No mask"
            cv2.putText(img, label_t + "  " + str(cnt_mask)+"  &"+" "+label_t_n + "  " + str(cnt_nomask), (int(img_w/2)-600, int(img_h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                        (255, 255, 255), 2)
            # cv2.putText(img, , (int(img_w/2), int(img_h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
            #             (255, 255, 255), 2)
            cv2.imshow('output', img)
            if FLAGS.output:
                print("done")
                out.write(img)


            logging.warning("Empty Frame")
            # time.sleep(0.1)
            count += 1
            if count < 10:
                print("None")
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        #ectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        # re.rectangle(img, (int(ret[0]), int(ret[1])), (int(ret[2]), int(ret[3])),(255,0,0), 2)
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        #
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            """logic for counting the real time people with mask and no mask """
            if class_name == "mask":
                mask_cnt_r = mask_cnt_r + 1
                mask_id.add(track.track_id)
                cnt_mask = len(mask_id)

            elif class_name == "nomask":
                Nomask_cnt_r = Nomask_cnt_r + 1
                nomask_id.add(track.track_id)
                cnt_nomask = len(nomask_id)

        label = "Mask   "
        label1 = "Nomask "
        cv2.rectangle(img, (0, 0), (400, 200), (0, 0, 0), -1)
        cv2.putText(img, label + "  " + str(mask_cnt_r), (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (255, 255, 255), 2)
        cv2.putText(img, label1 + "  " + str(Nomask_cnt_r), (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (255, 255, 255), 2)

        mask_cnt_r, Nomask_cnt_r = 0, 0  # for real time counting algorithm

        cv2.imshow('output', img)
        if FLAGS.output:
            print(FLAGS.output)
            out.write(img)
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
