import time
import warnings

warnings.filterwarnings("ignore")

from clean_data import *
from text2image_gan_ms import *
from Discriminator import Critic
from Generator import create_look_ahead_mask, create_padding_mask, Transformer

NUM_LAYERS = 4
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
BATCH_SIZE = 64
CRITIC_ITERATIONS = 2
LAMBDA = 10
TARGET_VOCAB_SIZE = 5000 + 1
DROPOUT_RATE = 0.1
ROW_SIZE = 8
COL_SIZE = 8
LATENT_DIMENSION = 100


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def i2T_dis_loss(f_cap, r_cap):
    b_shape = f_cap.shape[0]
    # f_label = tf.zeros([b_shape, 1, 1])
    # r_label = tf.ones([b_shape, 1, 1])
    r_cap = tf.reshape(r_cap, shape=(b_shape, 1, -1))
    r_output = i2T_critic(r_cap, True)
    # r_output = tf.reshape(r_output, shape=(b_shape))
    r_d_loss = loss_mse(tf.ones_like(r_output), r_output)
    r_d_loss = tf.reduce_sum(r_d_loss)

    f_cap = tf.reshape(f_cap, shape=(b_shape, 1, -1))
    f_output = i2T_critic(f_cap, True)
    # f_output = tf.reshape(f_output, shape=(b_shape))
    f_d_loss = loss_mse(tf.zeros_like(f_output), f_output)
    f_d_loss = tf.reduce_sum(f_d_loss)

    return r_d_loss + f_d_loss


def i2T_gen_loss(tar_real, predictions, f_cap, r_cap):
    loss = i2T_loss_function(tar_real, predictions)

    b_shape = f_cap.shape[0]
    r_cap = tf.reshape(r_cap, shape=(b_shape, 1, -1))
    g_output = i2T_critic(r_cap, True)
    # g_output = tf.reshape(g_output, shape=(b_shape))
    g_loss = loss_mse(tf.ones_like(g_output), g_output)
    g_loss = tf.reduce_sum(g_loss)

    return loss + g_loss


def i2T_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def calc_t2I_loss(real_captions, real_images):
    noise_1 = tf.random.normal([32, LATENT_DIMENSION])
    noise_2 = tf.random.normal([32, LATENT_DIMENSION])

    random_captions = generate_random_vectors(BATCH_SIZE)
    random_captions_1, random_captions_2 = tf.split(random_captions, 2, 0)
    real_captions_1, real_captions_2 = tf.split(real_captions, 2, 0)
    real_images_1, real_images_2 = tf.split(real_images, 2, 0)
    noise = tf.concat([noise_1, noise_2], 0)

    generated_images = t2I_generator([noise, real_captions], training=True)

    fake_1, fake_2 = tf.split(generated_images, 2, 0)

    f_fake_output_real_text_1 = t2I_discriminator(
        [fake_1, real_captions_1], training=True)
    f_fake_output_real_text_2 = t2I_discriminator(
        [fake_2, real_captions_2], training=True)

    r_real_output_real_text_1 = t2I_discriminator(
        [real_images_1, real_captions_1], training=True)
    r_real_output_real_text_2 = t2I_discriminator(
        [real_images_2, real_captions_2], training=True)

    f_real_output_fake_text_1 = t2I_discriminator(
        [real_images_1, random_captions_1], training=True)
    f_real_output_fake_text_2 = t2I_discriminator(
        [real_images_2, random_captions_2], training=True)

    gen_loss = t2I_generator_loss(
        f_fake_output_real_text_1) + t2I_generator_loss(f_fake_output_real_text_2)
    # mode seeking loss
    lz = tf.math.reduce_mean(tf.math.abs(
        fake_2 - fake_1)) / tf.math.reduce_mean(tf.math.abs(noise_2 - noise_1))

    eps = 1 * 1e-5
    loss_lz = 1 / (eps + lz) * ms_loss_weight
    total_gen_loss = gen_loss + loss_lz

    disc_loss_1 = t2I_discriminator_loss(r_real_output_real_text_1, f_fake_output_real_text_1,
                                         f_real_output_fake_text_1)
    disc_loss_2 = t2I_discriminator_loss(r_real_output_real_text_2, f_fake_output_real_text_2,
                                         f_real_output_fake_text_2)

    total_disc_loss = disc_loss_1 + disc_loss_2

    mse_images = tf.reduce_mean(loss_mse(generated_images, real_images))

    tf.print('t2I_D_loss:', [total_disc_loss], '  t2I_G_loss:', [total_gen_loss])

    return total_disc_loss, total_gen_loss, mse_images


# ###################################### TRAINING FUNCTIONS #########################################

@tf.function
def train_step(img_tensor, tar, img_name, img):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    dec_mask = create_masks_decoder(tar_inp)

    with tf.GradientTape() as tape, tf.GradientTape() as d_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as rnn:
        # predictions, _ = transformer(img_tensor, tar_inp, True, dec_mask)
        predictions, _ = i2T_generator(img_tensor, tar_inp, True, dec_mask)
        f_cap = tf.argmax(predictions, axis=-1)

        real_captions = text_to_zspace(f_cap)

        total_disc_loss, total_gen_loss, r_loss = calc_t2I_loss(real_captions, img)

        loss = i2T_gen_loss(tar_real, predictions, f_cap, tar_real)
        d_loss = i2T_dis_loss(f_cap, tar_real)

        loss += r_loss
        total_gen_loss += r_loss

    d_gradients = d_tape.gradient(d_loss, i2T_critic.trainable_variables)
    gradients = tape.gradient(loss, i2T_generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, t2I_discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(total_gen_loss, t2I_generator.trainable_variables)
    gradients_of_rnn = rnn.gradient(r_loss, text_to_zspace.trainable_variables)

    i2T_c_optimizer.apply_gradients(zip(d_gradients, i2T_critic.trainable_variables))
    i2T_g_optimizer.apply_gradients(zip(gradients, i2T_generator.trainable_variables))

    generator_optimizer.apply_gradients(zip(gradients_of_generator, t2I_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, t2I_discriminator.trainable_variables))
    rnn_optimizer.apply_gradients(zip(gradients_of_rnn, text_to_zspace.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train(dataset):
    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(t2I_generator_optimizer=generator_optimizer,
                                     t2I_discriminator_optimizer=discriminator_optimizer,
                                     t2I_generator=t2I_generator,
                                     t2I_discriminator=t2I_discriminator,
                                     i2T_generator_optimizer=i2T_g_optimizer,
                                     i2T_discriminator_optimizer=i2T_c_optimizer,
                                     i2T_generator=i2T_generator,
                                     i2T_discriminator=i2T_critic)

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for epoch in range(1):

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (img_tensor, tar, img_name, img)) in enumerate(dataset):
            train_step(img_tensor, tar, img_name, img)

            if (epoch + 1) % 10 == 0:
                [z_input, labels_input] = generate_latent_points(100, 25, cap_train)
                generate_and_save_images(t2I_generator,
                                         epoch + 1,
                                         [z_input, labels_input])

            if (epoch + 1) % 40 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(
                    epoch + 1, ckpt_save_path))

            if (epoch + 1) % 60 == 0:
                t2I_generator.save('Flick_text_to_image%03d.h5' % (epoch + 1))

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}')

        print(f'Epoch {epoch + 1}, Batch {batch}, Loss {train_loss.result()}, Accuracy {train_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch : {time.time() - start} secs\n')


# ################################  IMAGE2TEXT NETWORK AND OPTIMIZER ################################

learning_rate = CustomSchedule(D_MODEL)
i2T_generator = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, TARGET_VOCAB_SIZE,
                            max_pos_encoding=TARGET_VOCAB_SIZE, rate=DROPOUT_RATE)
i2T_critic = Critic(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF)

i2T_g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_mse = tf.keras.losses.BinaryCrossentropy(from_logits=True)
i2T_c_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

# ################################  TEXT2IMAGE NETWORK AND OPTIMIZER ################################
ms_loss_weight = 1.0
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000035, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000035, beta_1=0.5)
rnn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000035, beta_1=0.5)

t2I_discriminator = define_discriminator()
t2I_generator = define_generator()
text_to_zspace = TextEncode(TARGET_VOCAB_SIZE)

train(dataset)
