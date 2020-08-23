import pandas as pd
from keras.utils import to_categorical

from utils import *


def get_file_path(name):
    parsed = name.replace('.png', '').split('_')[1:]
    parsed = [int(n) for n in parsed]
    return f'data/raw/core50_350x350/s{parsed[0]}/o{parsed[1]}/{name}'


def preprocess_data():
    train = pd.read_csv('data/Core50/core50_train.csv')

    # COLUMN MODIFICATION
    train['Filename'] = train['Filename'].str.replace('jpg', 'png')
    train['class'] = train['class'] - 1
    train = train.assign(filepath=train['Filename'].apply(get_file_path))

    return train


def preprocess_and_save(sample=None):
    train = pd.read_csv('./data/raw/core50_train.csv')

    # COLUMN MODIFICATION
    train['filename'] = train['filename'].str.replace(
        'jpg', 'png').apply(get_file_path)
    train['class'] = train['class']

    if sample is not None:
        train = train.sample(sample)

    train.to_csv('data/train/tf_api_train/train.csv', index=False)

    return train


def gen_train_images(size=100):
    imgs_table = preprocess_data().sample(n=size)
    imgs_list = []
    class_list = []
    count = 0

    for index, row in imgs_table.iterrows():
        count += 1
        print(f'\rCount: {count} out of {len(imgs_table)}', end='')
        img = cv2.imread(str(row['filepath']), cv2.IMREAD_COLOR)
        box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        class_num = row['class']

        img = crop_and_resize(img, box)
        imgs_list.append(img)
        class_list.append(class_num)

    class_list = to_categorical(class_list, num_classes=50)
    print(class_list[0])
    np.save('../data/img_classifier_train/imgs_train.npy', np.array(imgs_list))
    np.save('../data/img_classifier_train/labels_train.npy', np.array(class_list))
    return imgs_list, class_list


def gen_test_dataset(starting_index=0, save_path='.'):
    train_raw = preprocess_data().loc[starting_index: starting_index + 300]
    test_dataset = []

    for index, row in train_raw.iterrows():
        box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        class_name = row['class']
        img = cv2.imread(str(row['filepath']), cv2.IMREAD_COLOR)
        test_dataset.append((img, box, class_name))

    np.save(f'{save_path}/test_4.npy', np.array(test_dataset))

    return test_dataset


def preprocess_box_time_series(starting_index=0, history_size=40):
    train = pd.read_csv(
        'data/raw/core50_train.csv')['xmin'].values

    x_dataset, y_dataset = time_series_segment(
        train, starting_index, starting_index + 300, history_size, 0, 1)

    for i in range(starting_index + 1, starting_index + 250):
        x_train_single, y_train_single = time_series_segment(train, i * 300, i * 300 + 300,
                                                             history_size, 0, 1)
        x_dataset.extend(x_train_single)
        y_dataset.extend(y_train_single)

    return np.asarray(x_dataset), np.asarray(y_dataset)


def time_series_segment(dataset, start_index, end_index, history_size,
                        target_size, step, single_step=True):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(dataset[i + target_size])
        else:
            labels.append(dataset[i:i + target_size])

    return data, labels


def main():
    SAVE_PATH = '../data/test_data'
    # gen_test_dataset(starting_index=36000, save_path=SAVE_PATH)
    # x_dataset, y_dataset = gen_train_images(size=35000)
    # preprocess_and_save(sample=1000)
    x, y = preprocess_box_time_series(0, 10)
    np.save('data/train/box_prediction_train/history_data.npy', x)
    np.save('data/train/box_prediction_train/target_data.npy', y)


if __name__ == "__main__":
    main()
