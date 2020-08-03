import cv2
import numpy as np


def show_img(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_and_resize(img, box, size=(75, 75)):
    box = [max(0, min(len(img[0]) - 1, int(point))) for point in box]
    img = img[box[1]:box[3], box[0]:box[2]]
    img = cv2.resize(img, size)
    return img


def overlay_box(img, box, prediction, color_BGR=(255, 0, 0), show=False):
    top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
    class_name, confidence = prediction
    text = f'{class_name}: {confidence:.3f}'

    img = cv2.rectangle(img, top_left, bottom_right, color_BGR, 1)
    img = cv2.putText(img, text, (top_left[0], top_left[1] - 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_BGR, 1)

    if show:
        show_img(img)

    return img


def overlay_boxes(img, boxes, color_BGR=(255, 0, 0), show=False):
    coordinates = [(tuple(box[:2]), tuple(box[2:])) for box in boxes]
    for coord in coordinates:
        top_left, bottom_right = coord
        img = cv2.rectangle(img, top_left, bottom_right, color_BGR, 1)

    if show:
        show_img(img)

    return img


def get_boxes(orig_box, margin=(0, 0), num=(1, 1), show=False, img=None):
    x_size, y_size = orig_box[2] - orig_box[0], orig_box[3] - orig_box[1]
    box = [orig_box[0] - margin[0],
           orig_box[1] - margin[1],
           orig_box[2] + margin[0],
           orig_box[3] + margin[1]]

    x_mins = np.linspace(box[0], box[2] - x_size, num=num[0], dtype=int)
    y_mins = np.linspace(box[1], box[3] - y_size, num=num[1], dtype=int)

    box_set = [orig_box]

    for x0 in x_mins:
        for y0 in y_mins:
            new_box = [x0, y0, x0 + x_size, y0 + y_size]
            if all(0 < pt < 350 for pt in new_box):
                box_set.append(new_box)

    if show:
        overlay_boxes(img, box_set, show=show)

    return np.asarray(box_set)


def compute_dist(box1, box2):
    return np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)


def map_to_class(prediction, class_names=None, class_text=False):
    if class_text:
        return class_names[np.argmax(prediction)], max(prediction)

    return np.argmax(prediction), max(prediction)


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]


def main():
    pass


if __name__ == '__main__':
    main()
