import pickle
from random import randint

import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import array_to_img
from numpy import asarray
from numpy import expand_dims
from numpy.random import randint
from tensorflow.keras import Model
from tensorflow.keras import layers


# model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
#                                                         binary=True)


def random_flip(image):
    image = tf.image.flip_left_right(image)
    return image.numpy()


def random_jitter(image):
    # add additional dimension necessary for zooming
    image = expand_dims(image, 0)
    image = image_augmentation_generator.flow(image, batch_size=1)
    # remove additional dimension (1, 64, 64, 3) to (64, 64, 3)
    result = image[0].reshape(image[0].shape[1:])
    return result


image_augmentation_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.8,
                                                                                           1.0])  # random zoom proves to be helpful in capturing more details https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/


def get_random_word_vectors_from_dataset(n_samples, captions):
    ix = np.random.randint(0, len(captions), n_samples)
    return np.asarray(captions)[ix]


def generate_random_vectors(n_samples):
    vectorized_random_captions = []
    for n in range(n_samples):
        vectorized_random_captions.append(
            tf.random.uniform([300]))
    return vectorized_random_captions


# Discriminator model
def define_discriminator():
    word_vector_dim = 300
    dropout_prob = 0.4

    in_label = layers.Input(shape=(300,))

    n_nodes = 3 * 64 * 64
    li = layers.Dense(n_nodes)(in_label)
    li = layers.Reshape((64, 64, 3))(li)

    dis_input = layers.Input(shape=(64, 64, 3))

    merge = layers.Concatenate()([dis_input, li])

    discriminator = layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding="same")(merge)
    discriminator = layers.LeakyReLU(0.2)(discriminator)
    discriminator = layers.GaussianNoise(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=64, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)

    discriminator = layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=128, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=256, kernel_size=(
        3, 3), strides=(2, 2), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same")(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Flatten()(discriminator)

    discriminator = layers.Dense(1024)(discriminator)

    discriminator = layers.LeakyReLU(0.2)(discriminator)

    discriminator = layers.Dense(1)(discriminator)

    discriminator_model = Model(
        inputs=[dis_input, in_label], outputs=discriminator)

    # discriminator_model.summary()

    return discriminator_model


def resnet_block(model, kernel_size, filters, strides):
    gen = model
    model = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)
    model = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = layers.Add()([gen, model])
    return model


# Generator model
def define_generator():
    kernel_init = tf.random_normal_initializer(stddev=0.02)
    batch_init = tf.random_normal_initializer(1., 0.02)

    random_input = layers.Input(shape=(100,))
    text_input1 = layers.Input(shape=(300,))
    text_layer1 = layers.Dense(8192)(text_input1)
    text_layer1 = layers.Reshape((8, 8, 128))(text_layer1)

    n_nodes = 128 * 8 * 8
    gen_input_dense = layers.Dense(n_nodes)(random_input)
    generator = layers.Reshape((8, 8, 128))(gen_input_dense)

    merge = layers.Concatenate()([generator, text_layer1])

    model = layers.Conv2D(filters=64, kernel_size=9,
                          strides=1, padding="same")(merge)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)

    gen_model = model

    for _ in range(4):
        model = resnet_block(model, 3, 64, 1)

    model = layers.Conv2D(filters=64, kernel_size=3,
                          strides=1, padding="same")(model)
    model = layers.BatchNormalization(momentum=0.5)(model)
    model = layers.Add()([gen_model, model])

    model = layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                   kernel_initializer=kernel_init)(model)
    model = layers.LeakyReLU(0.2)(model)

    model = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(model)

    generator_model = Model(inputs=[random_input, text_input1], outputs=model)

    # generator_model.summary()
    tf.keras.utils.plot_model(generator_model, to_file='model.png', show_shapes=True)
    return generator_model


def generate_latent_points(latent_dim, n_samples, captions):
    x_input = tf.random.normal([n_samples, latent_dim])
    text_captions = get_random_word_vectors_from_dataset(n_samples, captions)
    return [x_input, text_captions]


# Randomly flip some labels. Credits to https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
def noisy_labels(y, p_flip):
    n_select = int(p_flip * int(y.shape[0]))
    flip_ix = np.random.choice(
        [i for i in range(int(y.shape[0]))], size=n_select)

    op_list = []
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1.0, y[i]))
        else:
            op_list.append(y[i])

    outputs = tf.stack(op_list)
    return outputs


def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    print(predictions.shape)
    pyplot.figure(figsize=[7, 7])

    for i in range(predictions.shape[0]):
        pyplot.subplot(5, 5, i + 1)
        pyplot.imshow(array_to_img(predictions.numpy()[i]))
        pyplot.axis('off')

    pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # pyplot.show()


def t2I_discriminator_loss(r_real_output_real_text, f_fake_output_real_text_1, f_real_output_fake_text):
    alpha = 0.5
    real_output_noise = smooth_positive_labels(
        noisy_labels(tf.ones_like(r_real_output_real_text), 0.10))
    fake_output_real_text_noise_1 = smooth_negative_labels(
        tf.zeros_like(f_fake_output_real_text_1))
    real_output_fake_text_noise = smooth_negative_labels(
        tf.zeros_like(f_real_output_fake_text))

    real_loss = tf.reduce_mean(loss_mse(
        real_output_noise, r_real_output_real_text))
    fake_loss_ms_1 = tf.reduce_mean(loss_mse(
        fake_output_real_text_noise_1, f_fake_output_real_text_1))
    fake_loss_2 = tf.reduce_mean(loss_mse(
        real_output_fake_text_noise, f_real_output_fake_text))

    total_loss = real_loss + alpha * fake_loss_2 + (1 - alpha) * fake_loss_ms_1
    return total_loss


def t2I_generator_loss(f_fake_output_real_text):
    return tf.reduce_mean(loss_mse(tf.ones_like(f_fake_output_real_text), f_fake_output_real_text))


class TextEncode(tf.keras.Model):
    def __init__(self, vocab_size, out_dim=300):
        super().__init__()
        self.emb = layers.Embedding(input_dim=vocab_size, output_dim=out_dim)
        self.rnn = layers.LSTM(300)

    def call(self, x):
        x = self.emb(x)
        return self.rnn(x)


loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# @tf.function
# def train_step(images, epoch):
#     # define half_batch
#     latent_dim = 100
#     n_batch = 64
#
#     noise_1 = tf.random.normal([32, latent_dim])
#     noise_2 = tf.random.normal([32, latent_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as rnn:
#         real_captions = text_to_zspace(images[1])
#         real_images = images[3]
#
#         random_captions = generate_random_vectors(n_batch)
#         random_captions_1, random_captions_2 = tf.split(random_captions, 2, 0)
#         real_captions_1, real_captions_2 = tf.split(real_captions, 2, 0)
#         real_images_1, real_images_2 = tf.split(real_images, 2, 0)
#         noise = tf.concat([noise_1, noise_2], 0)
#
#         generated_images = generator([noise, real_captions], training=True)
#
#         fake_1, fake_2 = tf.split(generated_images, 2, 0)
#
#         f_fake_output_real_text_1 = discriminator(
#             [fake_1, real_captions_1], training=True)
#         f_fake_output_real_text_2 = discriminator(
#             [fake_2, real_captions_2], training=True)
#
#         r_real_output_real_text_1 = discriminator(
#             [real_images_1, real_captions_1], training=True)
#         r_real_output_real_text_2 = discriminator(
#             [real_images_2, real_captions_2], training=True)
#
#         f_real_output_fake_text_1 = discriminator(
#             [real_images_1, random_captions_1], training=True)
#         f_real_output_fake_text_2 = discriminator(
#             [real_images_2, random_captions_2], training=True)
#
#         #### Calculating losses ####
#
#         gen_loss = generator_loss(
#             f_fake_output_real_text_1) + generator_loss(f_fake_output_real_text_2)
#         # mode seeking loss
#         lz = tf.math.reduce_mean(tf.math.abs(
#             fake_2 - fake_1)) / tf.math.reduce_mean(tf.math.abs(noise_2 - noise_1))
#         eps = 1 * 1e-5
#         loss_lz = 1 / (eps + lz) * ms_loss_weight
#         total_gen_loss = gen_loss + loss_lz
#
#         tf.print('G_loss', [total_gen_loss])
#
#         disc_loss_1 = discriminator_loss(r_real_output_real_text_1, f_fake_output_real_text_1,
#                                          f_real_output_fake_text_1)
#         disc_loss_2 = discriminator_loss(r_real_output_real_text_2, f_fake_output_real_text_2,
#                                          f_real_output_fake_text_2)
#
#         total_disc_loss = disc_loss_1 + disc_loss_2
#
#         r_loss = tf.reduce_mean(loss_mse(generated_images, real_images))
#
#         tf.print('D_loss', [total_disc_loss])
#
#         #### Done calculating losses ####
#
#     gradients_of_discriminator = disc_tape.gradient(
#         total_disc_loss, discriminator.trainable_variables)
#
#     gradients_of_generator = gen_tape.gradient(
#         total_gen_loss, generator.trainable_variables)
#
#     gradients_of_rnn = rnn.gradient(
#         r_loss, text_to_zspace.trainable_variables)
#
#     generator_optimizer.apply_gradients(
#         zip(gradients_of_generator, generator.trainable_variables))
#
#     discriminator_optimizer.apply_gradients(
#         zip(gradients_of_discriminator, discriminator.trainable_variables))
#
#     rnn_optimizer.apply_gradients(
#         zip(gradients_of_rnn, text_to_zspace.trainable_variables)
#     )


# def train(dataset, epochs=500):
#     checkpoint_dir = 'checkpoints'
#     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#     checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                      discriminator_optimizer=discriminator_optimizer,
#                                      generator=generator,
#                                      discriminator=discriminator)
#
#     ckpt_manager = tf.train.CheckpointManager(
#         checkpoint, checkpoint_dir, max_to_keep=3)
#     if ckpt_manager.latest_checkpoint:
#         # ckpt_manager.checkpoints[3]
#         checkpoint.restore(ckpt_manager.latest_checkpoint)
#         print('Latest checkpoint restored!!')
#
#     for epoch in range(epochs):
#         start = time.time()
#         x = 0
#         for image_batch in dataset:
#             x = x + 1
#             print(x)
#             if (x < 126):
#                 train_step(image_batch, epoch)
#
#         if (epoch + 1) % 10 == 0:
#             [z_input, labels_input] = generate_latent_points(100, 25)
#             generate_and_save_images(generator,
#                                      epoch + 1,
#                                      [z_input, labels_input])
#
#         if (epoch + 1) % 40 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print('Saving checkpoint for epoch {} at {}'.format(
#                 epoch + 1, ckpt_save_path))
#
#         if (epoch + 1) % 60 == 0:
#             generator.save('Flick_text_to_image%03d.h5' % (epoch + 1))
#
#         print('Time for epoch {} is {} sec'.format(
#             epoch + 1, time.time() - start))


# ms_loss_weight = 1.0
#
# loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
# generator_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.000035, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.000035, beta_1=0.5)
# rnn_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.000035, beta_1=0.5)
#
# discriminator = define_discriminator()
# generator = define_generator()
# text_to_zspace = TextEncode(VOCAB_SIZE)

# images, lbs = load_data()
# BUFFER_SIZE = images.shape[0]
# BATCH_SIZE = 64

# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (images, lbs)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# '''  '''
# all_captions = []
# cap_file = './captions.pickle'
#
# if os.path.isfile(cap_file):
#     with open(cap_file, 'rb') as f:
#         all_captions = pickle.load(f)
# else:
#     raise NotImplemented
#
# all_img_name_vector = []
# img_name = './img_name.pickle'
#
# if os.path.isfile(img_name):
#     with open(img_name, 'rb') as f:
#         all_img_name_vector = pickle.load(f)
# else:
#     raise NotImplemented
#
#
# def load_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, (64, 64))
#     return img
#
#
# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<unk>',
#                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# tokenizer.fit_on_texts(all_captions)
#
# train_seqs = tokenizer.texts_to_sequences(all_captions)
# tokenizer.word_index['<pad>'] = 0
# train_seqs = tokenizer.texts_to_sequences(all_captions)
# cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
#
# img_name_train, img_name_val, cap_train, cap_val = train_test_split(all_img_name_vector, cap_vector, test_size=0.2,
#                                                                     random_state=0)
#
# num_steps = len(img_name_train)
#
#
# def map_func(img_name, cap):
#     img_tensor = np.load(img_name.decode('utf-8') + '.npy')
#     image = load_image(img_name.decode('utf-8'))
#     return img_tensor, cap, img_name, image
#
# # dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# # dataset = dataset.map(
# #     lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32, tf.string, tf.float32]),
# #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
# # dataset = dataset.shuffle(1000).batch(BATCH_SIZE)
# # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# # train(dataset)
