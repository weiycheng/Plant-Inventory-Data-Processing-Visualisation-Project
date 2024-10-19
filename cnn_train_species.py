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
directory_path = 'E:\\MAST90107\\cnn\\800'

buds_path = 'E:\\MAST90107\\cnn\\bud'
bark_path = 'E:\\MAST90107\\cnn\\bark'
stem_path = 'E:\\MAST90107\\cnn\\stem'
flower_path = 'E:\\MAST90107\\cnn\\flower'
fruit_path = 'E:\\MAST90107\\cnn\\fruit'
# 读取Excel文件
excel_file_path = 'E:\\MAST90107\\cnn\\train_data.xlsx'
data = pd.read_excel(excel_file_path)


# 加载图像并转换为数组
def load_and_preprocess_image(image_path):
    # 加载图像
    img = load_img(image_path, target_size=(img_height, img_width))
    # 将图像转换为数组
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 归一化像素值
    img_array /= 255.0
    return img_array

# 读取图像和标签
images = []
labels = []

# 遍历目录下的所有文件
for filename in os.listdir(directory_path):
    # 获取文件的完整路径
    file_path = os.path.join(directory_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        id = filename.split('_')[0][3:]
        match_row = data[data['objectID'] == int(id)]
        if match_row.size > 0:
            images.append(load_and_preprocess_image(file_path))
            labels.append(match_row['categories'].values[0])


for filename in os.listdir(buds_path):
    # 获取文件的完整路径
    file_path = os.path.join(buds_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        images.append(load_and_preprocess_image(file_path))
        labels.append('BUDS')

for filename in os.listdir(bark_path):
    # 获取文件的完整路径
    file_path = os.path.join(bark_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        images.append(load_and_preprocess_image(file_path))
        labels.append('BARK')

for filename in os.listdir(stem_path):
    # 获取文件的完整路径
    file_path = os.path.join(stem_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        images.append(load_and_preprocess_image(file_path))
        labels.append('STEM / TRUNK')


for filename in os.listdir(flower_path):
    # 获取文件的完整路径
    file_path = os.path.join(flower_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        images.append(load_and_preprocess_image(file_path))
        labels.append('FLOWERS')

for filename in os.listdir(fruit_path):
    # 获取文件的完整路径
    file_path = os.path.join(fruit_path, filename)
    # 检查是否为图片文件
    if os.path.isfile(file_path):
        images.append(load_and_preprocess_image(file_path))
        labels.append('FRUIT / CONES')

print(len(images))

# 转换列表为NumPy数组
images = np.array(images)
# 将标签转换为整数类型
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 保存 LabelEncoder 的类标签为 .npy 文件
np.save('E:\\MAST90107\\cnn\\label_classes.npy', label_encoder.classes_)

# 将标签转换为one-hot编码
num_classes = len(set(labels))
labels = to_categorical(labels, num_classes=num_classes)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# 加载基础模型

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False
# 创建模型
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
# 定义模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''

# 训练模型
epochs = 100
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)

# 评估模型的准确率
loss, accuracy = model.evaluate(images, labels)

print(f'Validation loss: {loss:.4f}')
print(f'Validation accuracy: {accuracy:.4f}')

# 保存模型
model.save('image_classifier_cwy.keras')

'''
# 预测验证集上的结果
y_pred = model.predict(X_val)

# 获取每个样本的预测类别
y_pred_classes = np.argmax(y_pred, axis=1)

# 获取预测类别与实际类别不一致的样本的索引
incorrect_indices = np.where(y_pred_classes != np.argmax(y_val, axis=1))[0]

# 获取这些样本的图像和标签
incorrect_images = X_val[incorrect_indices]
incorrect_labels = y_val[incorrect_indices]

# 获取这些样本的正确结果和错误结果
correct_labels = np.argmax(y_val, axis=1)
correct_labels = label_encoder.inverse_transform(correct_labels)
incorrect_labels = label_encoder.inverse_transform(y_pred_classes)

# 打印这些样本的正确结果和错误结果
for i in range(len(incorrect_indices)):
    print(f'Image {i+1}:')
    print(f'Correct label: {correct_labels[incorrect_indices[i]]}')
    print(f'Incorrect label: {incorrect_labels[incorrect_indices[i]]}')
    print()
'''