import time

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from utils import *


def main():
    frame_history = 40
    class_names = np.loadtxt('./data/classnames.txt',
                             delimiter='\n', dtype=np.str)
    print('Loaded class names.')

    dataset = np.load('./data/test/test_1.npy', allow_pickle=True)
    print('Loaded test dataset.')
    out = cv2.VideoWriter('./out/test_video_autobox_02.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (350, 350))

    seed_data, test_data = dataset[:frame_history], dataset[frame_history:]

    box_time_series = []
    class_time_series = []

    for frame in seed_data:
        img, box, class_val = frame
        img = crop_and_resize(img, box)
        img = overlay_box(img, box, (class_val, 1.0))

        box_time_series.append(np.array(box))
        class_time_series.append(class_val)
        out.write(img)

    print('Processed Seed Frames.')
    box_model = keras.models.load_model('./models/box_model_08_07_20.h5')
    box_model.trainable = False
    # box_model.summary()
    img_model = keras.models.load_model('./models/img_model_07_29_20.hdf5')
    img_model.trainable = False
    # img_model.summary()
    print('Loaded models.\nApplying algorithm to frames.')

    total_fps = 0
    count = 0

    for frame in test_data:
        start_time = time.time()
        last_time = time.time()
        checkpoints = []

        img = frame[0]

        input_boxes = np.expand_dims(
            np.asarray(box_time_series[-frame_history:]), axis=0).astype(np.float32)

        box_prediction = box_model.predict_on_batch(input_boxes)
        box_prediction = np.round(box_prediction)[0].astype(int)

        checkpoints.append(time.time() - last_time)
        last_time = time.time()

        box_list = get_boxes(box_prediction, margin=(8, 8), num=(4, 5))
        cropped_imgs = np.asarray(
            [crop_and_resize(img, box) for box in box_list])

        img_predictions = img_model.predict_on_batch(cropped_imgs)
        img_predictions = [map_to_class(pred) for pred in img_predictions]

        checkpoints.append(time.time() - last_time)

        common_class = mode(class_time_series[-frame_history:])

        img_predictions = [(pred[0], pred[1] + 0.1) if (pred[0] == common_class)
                           else pred for pred in img_predictions]

        pred_classes, pred_conf = zip(*img_predictions)

        best_boxes = box_list[np.where(pred_conf == max(pred_conf))]
        best_box = best_boxes[np.argmin([compute_dist(b, box_prediction)
                                         for b in best_boxes])]

        best_box_class = img_predictions[np.argmax(pred_conf)][0]

        img = overlay_box(img, best_box, (class_names[best_box_class],
                                          max(pred_conf)), color_BGR=(0, 0, 255))
        img = overlay_box(
            img, frame[1], (class_names[frame[2]], 1.0), color_BGR=(255, 0, 0))

        box_time_series.append(best_box)
        class_time_series.append(best_box_class)
        out.write(img)

        count += 1
        total_fps += (1.0 / (time.time() - start_time))
        print(
            f'\rStatus: {count} out of {len(test_data)}, FPS: {(1.0 / (time.time() - start_time)):6.4f}; {np.round(checkpoints, 3)}',
            end='')

    print(f'\nAverage FPS: {(total_fps / count):6.4f}')
    out.release()
    print('Video Saved.')


if __name__ == '__main__':
    main()
