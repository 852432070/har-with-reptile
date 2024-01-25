"""
Title: Few-Shot learning with Reptile
Author: [ADMoreau](https://github.com/ADMoreau)
Date created: 2020/05/21
Last modified: 2020/05/30
Description: Few-shot classification on the Omniglot dataset using Reptile.
Accelerator: GPU
"""

"""
## Introduction

The [Reptile](https://arxiv.org/abs/1803.02999) algorithm was developed by OpenAI to
perform model-agnostic meta-learning. Specifically, this algorithm was designed to
quickly learn to perform new tasks with minimal training (few-shot learning).
The algorithm works by performing Stochastic Gradient Descent using the
difference between weights trained on a mini-batch of never-seen-before data and the
model weights prior to training over a fixed number of meta-iterations.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Dropout
from tensorflow.keras import Model, regularizers

from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
## Define the Hyperparameters
"""

learning_rate = 1e-4
meta_step_size = 0.1

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 100000
eval_iters = 5
inner_iters = 4

eval_interval = 5
train_shots = 20
shots = 5
classes = 5

"""
## Prepare the data

The [Omniglot dataset](https://github.com/brendenlake/omniglot/) is a dataset of 1,623
characters taken from 50 different alphabets, with 20 examples for each character.
The 20 samples for each character were drawn online via Amazon's Mechanical Turk. For the
few-shot learning task, `k` samples (or "shots") are drawn randomly from `n` randomly-chosen
classes. These `n` numerical values are used to create a new set of temporary labels to use
to test the model's ability to learn a new task given few examples. In other words, if you
are training on 5 classes, your new class labels will be either 0, 1, 2, 3, or 4.
Omniglot is a great dataset for this task since there are many different classes to draw
from, with a reasonable number of samples for each class.
"""

data_path = '/data/wang_sc/datasets/PAMAP2_Dataset/Processed_self_made_unseen_activity_7/'

# %%
train_x = np.load(data_path + 'x_train.npy').astype(np.float32)
train_y = np.load(data_path + 'y_train.npy').astype(np.int32)
test_x = np.load(data_path + 'x_test.npy').astype(np.float32)
test_y = np.load(data_path + 'y_test.npy').astype(np.int32)

# train_x = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/x_train.npy').astype(np.float32)
# train_y = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/y_train.npy').astype(np.int32)
# test_x = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/x_test.npy').astype(np.float32)
# test_y = np.load('/data/wang_sc/datasets/PAMAP2_Dataset/Processed0/y_test.npy').astype(np.int32)


# # 将 train_x 和 test_x 沿着第一个轴（axis=0）进行合并
# merged_x = np.concatenate((train_x, test_x), axis=0)

# # 将 train_y 和 test_y 合并成一个数组
# merged_y = np.concatenate((train_y, test_y), axis=0)

# # 创建一个布尔掩码，标记标签值小于7的样本
# train_mask = merged_y < 7

# # 根据掩码将数据集划分为训练集和测试集
# train_x = merged_x[train_mask]
# train_y = merged_y[train_mask]
# test_x = merged_x[~train_mask]
# test_y = merged_y[~train_mask]

train_shape = train_x.shape
# train_x = train_x.reshape(train_shape[0], train_shape[1], train_shape[2], 1)
test_shape = test_x.shape
# test_x = test_x.reshape(test_shape[0], test_shape[1], test_shape[2], 1)
print(train_shape)
print(test_shape)
print(train_y.shape)
print(test_y.shape)

scaler = StandardScaler()
train_x = scaler.fit_transform(
train_x.astype(np.float32).reshape(-1,1)).reshape(train_shape[0], train_shape[1], train_shape[2], 1)
test_x = scaler.transform(
test_x.astype(np.float32).reshape(-1,1)).reshape(test_shape[0], test_shape[1], test_shape[2], 1)

# 将train_x中的元素按照train_y中的标签进行分类
x_dict = {}
for x, y in zip(train_x, train_y):
    if y not in x_dict:
        x_dict[y] = []
    x_dict[y].append(x)

# 打印每个标签对应的元素数量
for label, elements in x_dict.items():
    print('Label {}: {} elements'.format(label, len(elements)))
    # x_dict[label] = np.array(elements)

class Dataset:
    # This class will facilitate the creation of a few-shot dataset
    # from the Omniglot dataset that can be sampled from quickly while also
    # allowing to create new labels at the same time.
    def __init__(self, training):
        # Download the tfrecord files containing the omniglot data and convert to a
        # dataset.
        # split = "train" if training else "test"
        # ds = tfds.load("omniglot", data_dir='./dataset', split=split, as_supervised=True, shuffle_files=False, download=False)
        # Iterate over the dataset to get each individual image and its class,
        # and put that data into a dictionary.
        self.data = {}

        def extraction(image, label):
            # This function will shrink the Omniglot images to the desired size,
            # scale pixel values and convert the RGB image to grayscale
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            return image, label

        for i in range(train_x.shape[0]):
            image = train_x[i]
            label = str(train_y[i])
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(image)
        self.labels = list(self.data.keys())

    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, split=False
    ):
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 171, 9, 1))
        if split:           
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 171, 9, 1))

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx    #给每个label赋class_idx初值
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        if split:
            return dataset, test_images, test_labels
        return dataset
    


import urllib3

urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

# """
# ## Visualize some examples from the dataset
# """

# _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

# sample_keys = list(train_dataset.data.keys())

# for a in range(5):
#     for b in range(5):
#         temp_image = train_dataset.data[sample_keys[a]][b]
#         temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
#         temp_image *= 255
#         temp_image = np.clip(temp_image, 0, 255).astype("uint8")
#         if b == 2:
#             axarr[a, b].set_title("Class : " + sample_keys[a])
#         axarr[a, b].imshow(temp_image, cmap="gray")
#         axarr[a, b].xaxis.set_visible(False)
#         axarr[a, b].yaxis.set_visible(False)
# plt.show()

"""
## Build the model
"""


# def conv_bn(x):
#     x = layers.Conv2D(filters=64, kernel_size=(3,1), strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     return layers.ReLU()(x)


# inputs = layers.Input(shape=(171, 9, 1))
# x = conv_bn(inputs)
# x = conv_bn(x)
# x = conv_bn(x)
# x = conv_bn(x)
# x = layers.Flatten()(x)
# outputs = layers.Dense(classes, activation="softmax")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile()
# optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

#标准卷积块
def conv_block(
    inputs,
    filters,
    kernel_size=(3,1),
    strides=(1,1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.01), strides=strides, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    return tf.keras.layers.ReLU(6.0)(x)
##深度可分离卷积块
def depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    strides=(1,1),
    expansion=4
):
    input_channel = inputs.shape[-1]

    x = tf.keras.layers.Conv2D(input_channel * expansion, kernel_size=(1,1), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(input_channel * expansion,(6, 1), padding='same', strides=strides, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    ###深度卷积到此结束



    
    ###下面是逐点卷积
    x = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1,1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.2)(x)
    
    atte = tf.keras.layers.GlobalAveragePooling2D()(x)
    atte = Dense(pointwise_conv_filters, activation='relu')(atte)
    print("atte shape")
    print(atte.shape)

    x = tf.keras.layers.Multiply()([x, atte])
    # avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=[1, 3], keepdims=True))(x)
    # print("avgpool shape")
    # print(avg_pool.shape)
    # max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=[1, 3], keepdims=True))(x)
    # concat = tf.keras.layers.Concatenate(axis=2)([avg_pool, max_pool])
    # concat = Flatten()(concat)
    # print("concat shape")
    # print(concat.shape)
    # # atte = tf.keras.layers.Conv2D(filters = 1, kernel_size=(6,1), padding='same', use_bias=False)(concat)
    atte_c2 = Dense(9, activation='relu')(atte)
    # atte = Dense(9, activation='relu')(concat)
    
    x = tf.keras.layers.Multiply()([x, tf.expand_dims(atte_c2, axis=2)])


    # x = tf.keras.layers.Multiply()([x, tf.expand_dims(atte, axis=2)])


    identity = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1,1), padding='same', strides=strides, use_bias=False, activation='sigmoid')(inputs)
    identity = tf.keras.layers.BatchNormalization()(identity)
    x = tf.keras.layers.ReLU()(x)
    identity = Dropout(0.2)(identity)
    
    return tf.keras.layers.add([x,identity])
 
#mobile_net
def mobilenet_v1(
    inputs,
    classes
):
    channel_size = 8
    ##特征提取层
    x = conv_block(inputs, channel_size, strides=(2,1))
#     x = depthwise_conv_block(x, 64)
#     x = depthwise_conv_block(x, 64, strides=(2,1))
    # x = depthwise_conv_block(x, channel_size*2, strides=(2,1))
#     x = depthwise_conv_block(x, 128)
#     x = depthwise_conv_block(x, 128, strides=(2,1))
    x = depthwise_conv_block(x, channel_size*4, strides=(2,1))
#     x = depthwise_conv_block(x, 256)
#     x = depthwise_conv_block(x, 256, strides=(2,1))
#     x = depthwise_conv_block(x, 256)
    # x = depthwise_conv_block(x, channel_size*8, strides=(2,1))
    x = depthwise_conv_block(x, channel_size*16)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 512)
#     x = depthwise_conv_block(x, 1024)
    # x = depthwise_conv_block(x, channel_size*32)
#     x = depthwise_conv_block(x, 1024, strides=(2,1))
#     x = depthwise_conv_block(x, channel_size)
    
    ##全局池化
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    ##全连接层
    pred = tf.keras.layers.Dense(classes, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(x)
    
    return pred
 
 
##模型实例化
inputs = tf.keras.Input(shape=(171,9,1))
model = tf.keras.Model(inputs=inputs, outputs=mobilenet_v1(inputs, classes))
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
"""
## Train the model
"""

import math

restart_periods = [10000, 20000, 30000, 40000]  # 重启周期列表

# 定义余弦退火函数
def cosine_annealing_with_restarts(current_step, restart_periods, initial_lr):
    # 计算当前周期和周期内的步骤
    period_index = 0
    while period_index < len(restart_periods) and current_step >= restart_periods[period_index]:
        current_step -= restart_periods[period_index]
        period_index += 1

    # 如果在周期列表范围外，则使用最后一个周期
    current_period = restart_periods[min(period_index, len(restart_periods) - 1)]
    
    # 调整学习率
    return initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / current_period))

training = []
testing = []
for meta_iter in range(meta_iters):
    loss_sum = 0
    loss_num = 0
    frac_done = meta_iter / meta_iters
    cur_meta_step_size = cosine_annealing_with_restarts(meta_iter, restart_periods, meta_step_size)
    # cur_meta_step_size = cosine_annealing(meta_iter, meta_iters, meta_step_size)
    # Temporarily save the weights from the model.
    old_vars = model.get_weights()
    # Get a sample from the full dataset.
    mini_dataset = train_dataset.get_mini_dataset(
        inner_batch_size, inner_iters, train_shots, classes
    )
    for images, labels in mini_dataset:
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
            loss_sum += sum(loss)
            loss_num += len(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    new_vars = model.get_weights()
    print(f"iter: {meta_iter}, loss: {loss_sum/loss_num}")
    # Perform SGD for the meta step.
    for var in range(len(new_vars)):
        new_vars[var] = old_vars[var] + (
            (new_vars[var] - old_vars[var]) * cur_meta_step_size
        )
    # After the meta-learning step, reload the newly-trained weights into the model.
    model.set_weights(new_vars)
    # Evaluation loop
    if meta_iter % eval_interval == 0:
        loss_sum = 0
        loss_num = 0
        accuracies = []
        for dataset in (train_dataset, test_dataset):
            # Sample a mini dataset from the full dataset.
            train_set, test_images, test_labels = dataset.get_mini_dataset(
                eval_batch_size, eval_iters, shots, classes, split=True
            )
            old_vars = model.get_weights()
            # Train on the samples and get the resulting accuracies.
            for images, labels in train_set:
                with tf.GradientTape() as tape:
                    preds = model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                    loss_sum += sum(loss)
                    loss_num += len(loss)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            test_preds = model.predict(test_images)
            test_preds = tf.argmax(test_preds).numpy()
            num_correct = (test_preds == test_labels).sum()
            # Reset the weights after getting the evaluation accuracies.
            model.set_weights(old_vars)
            accuracies.append(num_correct / classes)
        print(f"eval: {meta_iter}, loss: {loss_sum/loss_num}")
        training.append(accuracies[0])
        testing.append(accuracies[1])
        if meta_iter % 100 == 0:
            print(
                "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
            )

"""
## Visualize Results
"""

# # First, some preprocessing to smooth the training and testing arrays for display.
# window_length = 100
# train_s = np.r_[
#     training[window_length - 1 : 0 : -1], training, training[-1:-window_length:-1]
# ]
# test_s = np.r_[
#     testing[window_length - 1 : 0 : -1], testing, testing[-1:-window_length:-1]
# ]
# w = np.hamming(window_length)
# train_y = np.convolve(w / w.sum(), train_s, mode="valid")
# test_y = np.convolve(w / w.sum(), test_s, mode="valid")

# Display the training accuracies.
x = np.arange(0, len(test_y), 1)
plt.plot(x, test_y, x, train_y)
plt.legend(["test", "train"])
plt.grid()
#获取一个mini_dataset
train_set, test_images, test_labels = dataset.get_mini_dataset(
    eval_batch_size, eval_iters, shots, classes, split=True
)
for images, labels in train_set:
    with tf.GradientTape() as tape:
        preds = model(images)
        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
test_preds = model.predict(test_images)
test_preds = tf.argmax(test_preds).numpy()

_, axarr = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())

for i, ax in zip(range(5), axarr):
    temp_image = np.stack((test_images[i, :, :, 0],) * 3, axis=2)
    temp_image *= 255
    temp_image = np.clip(temp_image, 0, 255).astype("uint8")
    ax.set_title(
        "Label : {}, Prediction : {}".format(int(test_labels[i]), test_preds[i])
    )
    ax.imshow(temp_image, cmap="gray")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
plt.show()