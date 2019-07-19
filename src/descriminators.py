#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (Dense, Input, LeakyReLU)
from keras.models import Model, Sequential
from keras.optimizers import Adam

"""
Most of this based on tutorial about GANs 
(https://towardsdatascience.com/generative-adversarial-networks-gans-a-beginners-guide-5b38eceece24)
"""


def get_discriminator(input_shape=(256,)):
    model = Sequential()
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    disc_inp = Input(input_shape)
    validity = model(disc_inp)

    return Model(disc_inp, validity)


def get_generator(model_name):
    """
    Loads pretrained model (encoder)
    """
    pass


def get_GAN(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    generator.compile(loss='rmse',
                      optimizer=optimizer)

    inp = Input(batch_shape=(128, 500, 3))
    feautres = generator(inp)
    discriminator.trainable = False

    valid = discriminator(feautres)
    combined = Model(inp, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    return combined


# Train
epochs = 100
batch_size = 128

callbacks_combined = [ModelCheckpoint('best_gan', save_best_only=True, monitor='loss', mode='min', verbose=1)]
callbacks_discriminator = [
    ModelCheckpoint('best_discriminator', save_best_only=True, monitor='loss', mode='min', verbose=1)]

generator = get_generator()
discriminator = get_discriminator()
combined = get_GAN(generator, discriminator)

for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    X_sensor = get_sensor_values()
    X_fake = generator.predict(X_sensor)
    X_real = get_features(f"{sensor}2{sensor}_features")

    y_fake = np.zeros(len(X_fake))
    y_real = np.ones(len(X_real))

    X = np.concat((X_fake, X_real))
    Y = np.concat((y_fake, y_real))
    d_loss = discriminator.train(X, Y,
                                 callbacks=callbacks_discriminator,
                                 shuffle=True)

    # ---------------------
    #  Train Generator
    # ---------------------
    # The generator wants the discriminator to label the generated samples
    # as valid (ones)
    valid_y = np.ones(len(X_fake))
    g_loss = combined.train(X_sensor, valid_y,
                            callbacks=callbacks_combined,
                            shuffle=True)

    # Plot the progress
    print("Epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
