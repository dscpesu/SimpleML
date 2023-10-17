import pandas as pd
import numpy as np
from PIL import Image

train = pd.read_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\csv_train.csv')
test = pd.read_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\csv_test.csv')

train = train.drop(columns=['width', 'height', 'class'])
test = test.drop(columns=['width', 'height', 'class'])

train = train.groupby('filename').agg(lambda x: tuple(x)).reset_index()
test = test.groupby('filename').agg(lambda x: tuple(x)).reset_index()

def preprocess_y(Y, orig_img_shape=(600, 900, 3), output_shape=(8, 12, 5)):
    outputs = np.zeros(Y.shape[0:1] + output_shape)
    for i, y in enumerate(Y):
        if isinstance(y, str):
            y = eval(y)
        for box in y:
            grid_x = orig_img_shape[1] / output_shape[1]
            grid_y = orig_img_shape[0] / output_shape[0]
            h = int(box[0] / grid_y)
            w = int(box[1] / grid_x)
            box[0] %= grid_y
            box[1] %= grid_x
            box[0] /= grid_y
            box[1] /= grid_x
            box[2] /= grid_y
            box[3] /= grid_x
            outputs[i, h, w] = np.array([1] + box)
    return outputs

def save_y():
    def corners_to_boxes(corners):
        xmin, xmax, ymin, ymax = corners
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        return [y, x, height, width]

    train_y = train.apply(lambda row: zip(row['xmin'], row['xmax'], row['ymin'], row['ymax']), axis=1)
    train_y = train_y.apply(lambda x: tuple(map(corners_to_boxes, x)))

    test_y = test.apply(lambda row: zip(row['xmin'], row['xmax'], row['ymin'], row['ymax']), axis=1)
    test_y = test_y.apply(lambda x: tuple(map(corners_to_boxes, x)))

    train_y.to_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\orig_train_y.csv', index=False)
    test_y.to_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\orig_test_y.csv', index=False)

    train_y = preprocess_y(train_y)
    test_y = preprocess_y(test_y)

    np.save(r'Urdu_Text_Detection\Dataset\archive\UrTiV\train_y', train_y)
    np.save(r'Urdu_Text_Detection\Dataset\archive\UrTiV\test_y', test_y)

def save_x():
    def link_to_pixels(link, base_url):
        try:
            link = base_url + '\\' + link
            im = Image.open(link).resize((96, 64))
            pixels = np.array(im)
            return pixels
        except:
            return None

    # global train, test

    # train = train[0:100]
    base_url = r'Urdu_Text_Detection\Dataset\archive\UrTiV\Train'
    train_x = train['filename'].apply(lambda x: link_to_pixels(x, base_url))
    train_x = np.stack(train_x, axis=0)
    train_x = train_x.astype(np.float32) / 255

    # test = test[0:100]
    base_url = r'Urdu_Text_Detection\Dataset\archive\UrTiV\Test'
    test_x = test['filename'].apply(lambda x: link_to_pixels(x, base_url))
    test_x = np.stack(test_x, axis=0)
    test_x = test_x.astype(np.float32) / 255

    np.save(r'Urdu_Text_Detection\Dataset\archive\UrTiV\train_x', train_x)
    np.save(r'Urdu_Text_Detection\Dataset\archive\UrTiV\test_x', test_x)

def load_data():
    orig_train_y = pd.read_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\orig_train_y.csv')['0']
    orig_test_y = pd.read_csv(r'Urdu_Text_Detection\Dataset\archive\UrTiV\orig_test_y.csv')['0']

    train_y = np.load(r'Urdu_Text_Detection\Dataset\archive\UrTiV\train_y.npy')
    test_y = np.load(r'Urdu_Text_Detection\Dataset\archive\UrTiV\test_y.npy')

    train_x = np.load(r'Urdu_Text_Detection\Dataset\archive\UrTiV\train_x.npy')
    test_x = np.load(r'Urdu_Text_Detection\Dataset\archive\UrTiV\test_x.npy')

    return train_x, train_y, orig_train_y, test_x, test_y, orig_test_y

if __name__ == '__main__':
    save_x()
    save_y()
    train_x, train_y, orig_train_y, test_x, test_y, orig_test_y = load_data()
    print(train_x.shape, train_y.shape, orig_train_y.shape)