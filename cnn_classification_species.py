import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import Sequential
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.saving import load_model
from keras.src.utils import load_img, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 设置图像尺寸
img_height, img_width = 150, 150

# 图片目录
directory_path = 'E:\\MAST90107\\export_photo'


# 加载图像并转换为数组
def load_and_preprocess_image(image_path):
    # 加载图像
    img = load_img(image_path, target_size=(img_height, img_width))
    # 将图像转换为数组
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 归一化像素值
    img_array /= 255.0
    return img_array

def convert_predictions_to_labels(class_indices, class_mapping):
    labels = [class_mapping[idx] for idx in class_indices]
    return labels

# 读取图像和标签
images = []
ids = []

# 遍历目录下的所有文件
for filename in os.listdir(directory_path):
    print(filename)
    # 获取文件的完整路径
    file_path = os.path.join(directory_path, filename)
    # 检查是否为图片文件
    try:
        if filename.endswith('jpg') or filename.endswith('jpeg'):
            id = filename.split('_')[0][3:]
            images.append(load_and_preprocess_image(file_path))
            ids.append(id)
    except Exception as e:
        print(e)

class_mapping = {
    4: 'LEAVES / FOLIAGE',
    5: 'STEM / TRUNK',
    1: 'BUDS',
    2: 'FLOWERS',
    0: 'BARK',
    3: 'FRUIT / CONES'
}

# 转换列表为NumPy数组
images = np.array(images)

# 加载模型
model = load_model('image_classifier_cwy.keras')
predictions = model.predict(images)
y_pred_classes = np.argmax(predictions, axis=1)
labels = convert_predictions_to_labels(y_pred_classes, class_mapping)


# 创建一个DataFrame
df = pd.DataFrame({
    'objectID': ids,
    'categories': labels
})

# 保存为Excel文件
output_file = 'model_species.xlsx'
df.to_excel(output_file, index=False)