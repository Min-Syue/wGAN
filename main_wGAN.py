from keras.datasets import mnist, cifar10
from model.wGAN import wGAN
from model.funs import find_diff_label
from keras.optimizers import Adam, RMSprop

import keras
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
0:airplane
1:automobile
2:bird
3:cat
4:deer
5:dog
6:frog
7:horse
8:ship
9:truck
'''

'''
add some note
'''

if __name__ == '__main__':

    #add some notation
    # 在mac上做github測試

    # 設定每個隱向量的大小
    latent_size_tp = 100

    # 設定訓練時的參數
    batch_size = 32
    half_batch_size = int(batch_size / 2)
    epochs = 4000

    # 宣告追蹤的數值之圖變化
    tracing_size = 49
    test_vector = tf.random.normal([tracing_size, latent_size_tp])

    # 宣告儲存圖片的名字
    my_path = os.path.abspath(os.path.dirname(__file__))
    Images_name = my_path + '/photo_wGAN/photo_mnist/Images_Epochs_'

    # 宣告評斷模型要訓練幾次
    take_size = 5

    # 宣告模型
    model = wGAN(opt_g=RMSprop(learning_rate=0.00005), opt_d=RMSprop(learning_rate=0.00005), latent_size=latent_size_tp, channels=1, width=28, height=28)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    concat_x = np.concatenate([X_train, X_test], axis = 0)
    
    output_train = concat_x

    all_images = (output_train.astype("float32") - 127.5) / 127.5
    # all_images = output_train.astype("float32") - 255.0
    all_images = np.reshape(all_images, (-1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices((all_images))

    plt.figure(figsize=(12,12))

    for e in range(epochs):

        d_loss_final = 0

        for itr, data in enumerate(dataset.shuffle(1024).batch(batch_size).take(take_size)):

            d_loss = model.train_d_model_step(data, batch_size=batch_size, clip_min=-0.01, clip_max=0.01)
            d_loss_final += d_loss
            
        g_loss = model.train_g_model_step(batch_size=batch_size * 2)
        
        print ('nEpochs: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (e+1, d_loss_final / take_size, g_loss))
        
        # 追蹤生成圖片
        trace_save_image = model.Generate_model.predict(test_vector)
        trace_save_image = np.squeeze(trace_save_image)

        for image_idx in range(tracing_size):
            plt.subplot(7, 7, image_idx+1) 
            plt.imshow(trace_save_image[image_idx], cmap='gray')
            plt.title("label %d" % (image_idx+1))

        plt.tight_layout()
        plt.savefig(Images_name+str(e+1))
