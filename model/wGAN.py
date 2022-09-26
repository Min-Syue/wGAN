from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, UpSampling2D, ZeroPadding2D
from keras.layers import LeakyReLU, BatchNormalization, ReLU

import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

class wGAN:
    def __init__(self , opt_g, opt_d, latent_size=100 ,width=28, height=28, channels=1):
    
        # 定義各種參數
        self._latent_size = latent_size
        self._width = width
        self._height = height
        self._channels = channels

        # 定義 loss function
        self.loss_fn = wasserstein_loss

        # 定義優化器
        self.optimizer_g = opt_g
        self.optimizer_d = opt_d

        # 宣告生成網路
        self.Generate_model = self._bulit_generate()
        self.Generate_model.summary()

        # 宣告判別真假網路
        self.Discriminator_model = self._bulit_Discri()
        self.Discriminator_model.summary()

    def _bulit_generate(self):

        input = Input(shape=(self._latent_size, ))
        d1 = Dense(7 * 7 * 128)(input)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1_reshape = Reshape((7, 7, 128))(d1)

        Conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(d1_reshape)
        Conv1 = BatchNormalization()(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)

        Conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(Conv1)
        Conv1 = BatchNormalization()(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)

        out_image = Conv2D(1, (7, 7), padding="same", activation="tanh")(Conv1)

        generator = tf.keras.Model(inputs=input, outputs=out_image)

        return generator

        '''
        input = Input(shape=(self._latent_size, ))
        d1 = Dense(7 * 7 * 128)(input)
        d1 = ReLU()(d1)
        d1 = Reshape((7, 7, 128))(d1)

        s1 = UpSampling2D()(d1)
        Conv1 = Conv2D(128, kernel_size=(4, 4), padding='same')(s1)
        Conv1 = BatchNormalization(momentum=0.8)(Conv1)
        Conv1 = ReLU()(Conv1)

        s1 = UpSampling2D()(Conv1)
        Conv1 = Conv2D(64, kernel_size=(4,4), padding="same")(s1)
        Conv1 = BatchNormalization(momentum=0.8)(Conv1)
        Conv1 = ReLU()(Conv1)

        out_image = Conv2D(self._channels, kernel_size=(4,4), padding="same", activation="tanh")(Conv1)

        generator = tf.keras.Model(inputs=input, outputs=out_image)

        return generator
        '''
    def _bulit_Discri(self):
        
    
        input = Input(shape=(self._width, self._height, self._channels))

        Conv1 = Conv2D(64, (4, 4), strides=(2,2), padding="same")(input)
        Conv1 = BatchNormalization()(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)

        Conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(Conv1)
        Conv1 = BatchNormalization()(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)

        Flat = Flatten()(Conv1)
        out_image = Dense(1)(Flat)

        discrimintor = tf.keras.Model(inputs=input, outputs=out_image)

        return discrimintor
        
        '''
        input = Input(shape=(self._width, self._height, self._channels))
        Conv1 = Conv2D(16, kernel_size=3, strides=2, padding="same")(input)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)
        Conv1 = Dropout(0.25)(Conv1)

        Conv1 = Conv2D(32, kernel_size=3, strides=2, padding="same")(Conv1)
        Conv1 = ZeroPadding2D(padding=((0,1),(0,1)))(Conv1)
        Conv1 = BatchNormalization(momentum=0.8)(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)
        Conv1 = Dropout(0.25)(Conv1)

        Conv1 = Conv2D(64, kernel_size=3, strides=2, padding="same")(Conv1)
        Conv1 = BatchNormalization(momentum=0.8)(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1)
        Conv1 = Dropout(0.25)(Conv1)

        Conv1 = Conv2D(128, kernel_size=3, strides=1, padding="same")(Conv1)
        Conv1 = BatchNormalization(momentum=0.8)(Conv1)
        Conv1 = LeakyReLU(alpha=0.2)(Conv1) 
        Conv1 = Dropout(0.25)(Conv1)

        Conv1 = Flatten()(Conv1)
        out_image = Dense(1)(Conv1)

        discrimintor = tf.keras.Model(inputs=input, outputs=out_image)

        return discrimintor
        '''
        
    @tf.function
    def train_d_model_step(self, data, batch_size=32, clip_min=-2.0, clip_max=2):
        
        # data已經在外圍做好隨機挑選(這邊是已經隨機選出來的data)
        '''
        # 先train real images

        real_labels = -tf.ones([batch_size, 1])

        with tf.GradientTape() as tape:
            predict_labels = self.Discriminator_model(data)
            d_loss_r = self.loss_fn(real_labels, predict_labels)
        grads = tape.gradient(d_loss_r, self.Discriminator_model.trainable_weights)
        self.optimizer_d.apply_gradients(zip(grads, self.Discriminator_model.trainable_weights))

        # 再來train fake iamges

        # 生成隨機的向量和隨機的圖片
        gen_noise = tf.random.normal([batch_size, self._latent_size])

        fake_labels = tf.ones([batch_size, 1])

        with tf.GradientTape() as tape:
            predict_labes = self.Discriminator_model(self.Generate_model(gen_noise))
            d_loss_f = self.loss_fn(fake_labels, predict_labes)
        grads = tape.gradient(d_loss_f, self.Discriminator_model.trainable_weights)
        self.optimizer_d.apply_gradients(zip(grads, self.Discriminator_model.trainable_weights))

        for weight in self.Discriminator_model.trainable_variables:
            weight.assign(tf.clip_by_value(weight, clip_min, clip_max))


        return (d_loss_r + d_loss_f) * 0.5
        '''
        
        # data已經在外圍做好隨機挑選(這邊是已經隨機選出來的data)
        
        # 生成隨機的向量和隨機的圖片
        gen_noise = tf.random.normal([batch_size, self._latent_size])
        gen_images = self.Generate_model(gen_noise)
        gen_noise_labels = tf.concat(
            [data, gen_images], axis=0
        )

        y_combined_data = tf.concat([
            -tf.ones([batch_size, 1]), tf.ones([batch_size, 1])], 0
            )

        # 訓練discrimintor model
        with tf.GradientTape() as tape:
            predict_labes = self.Discriminator_model(gen_noise_labels)
            d_loss = self.loss_fn(y_combined_data, predict_labes)
        grads = tape.gradient(d_loss, self.Discriminator_model.trainable_weights)
        self.optimizer_d.apply_gradients(zip(grads, self.Discriminator_model.trainable_weights))

        for weight in self.Discriminator_model.trainable_variables:
            weight.assign(tf.clip_by_value(weight, clip_min, clip_max))

        return d_loss
        

    @tf.function
    def train_g_model_step(self, batch_size=32):

        # 生成隨機的向量
        gen_noise = tf.random.normal([batch_size, self._latent_size])
        y_fake_label = -tf.ones([batch_size, 1])

        # 訓練generate model
        with tf.GradientTape() as tape:
            fake_images = self.Generate_model(gen_noise)
            predict_labes = self.Discriminator_model(fake_images)
            g_loss = self.loss_fn(y_fake_label, predict_labes)
        grads = tape.gradient(g_loss, self.Generate_model.trainable_weights)
        self.optimizer_g.apply_gradients(zip(grads, self.Generate_model.trainable_weights))

        return g_loss
