from PIL import Image, ImageDraw

import loader
import model as Model

model = Model.load('saved_model')
train_x, train_y, orig_train_y, test_x, test_y, orig_test_y = loader.load_data()

yhat = model.predict(test_x)

base_url = r'Urdu_Text_Detection\Dataset\archive\UrTiV\Test'
for i in range(0, 600, 60):
    link = loader.test['filename'][i]
    image = Image.open(base_url + '\\' + link)
    draw = ImageDraw.Draw(image)

    scores, boxes = Model.yolo_eval(yhat[i], img_shape=(600,900,3), score_threshold=0.2, iou_threshold=0.2)
    corners = Model.yolo_boxes_to_corners(boxes).numpy()
    corners = corners[:, [1, 0, 3, 2]]

    for j in corners:
        draw.rectangle(j, outline='green', width=5)
    
    image.show()