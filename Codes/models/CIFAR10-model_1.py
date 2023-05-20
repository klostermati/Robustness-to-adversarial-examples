# Training parameters
batch_size = 32
nb_epoch = 50
nb_filters = 32
pool_size = (3, 3)
kernel_size = (3, 3)

# Model
training = False
if(not ('model' in locals())):
    training = True
    print("Creating model")

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, padding='same', input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(tf.keras.layers.Conv2D(2*nb_filters, kernel_size, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(2*nb_filters, kernel_size))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(tf.keras.layers.Conv2D(int(nb_filters/2), kernel_size, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(nb_classes))
    model.add(tf.keras.layers.Activation('softmax'))

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
            featurewise_center=False,               # set input mean to 0 over the dataset
            samplewise_center=False,                # set each sample mean to 0
            featurewise_std_normalization=False,    # divide inputs by std of the dataset
            samplewise_std_normalization=False,     # divide each input by its std
            zca_whitening=False,                    # apply ZCA whitening
            zca_epsilon=1e-06,                      # epsilon for ZCA whitening
            rotation_range=0,                       # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
            shear_range=0.,                         # set range for random shear
            zoom_range=0.,                          # set range for random zoom
            channel_shift_range=0.,                 # set range for random channel shifts
            fill_mode='nearest',                    # set mode for filling points outside the input boundaries
            cval=0.,                                # value used for fill_mode = "constant"
            horizontal_flip=True,                   # randomly flip images
            vertical_flip=False,                    # randomly flip images
            rescale=None,                           # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,            # set function that will be applied on each input
            data_format=None,                       # image data format, either "channels_first" or "channels_last"
            validation_split=0.0)                   # fraction of images reserved for validation (strictly between 0 and 1)
    
    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    history=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=nb_epoch, verbose=verbose, steps_per_epoch=X_train.shape[0]/batch_size,validation_data=(X_test, Y_test))

# Model 1
if((not training) and args.detection == "stoGauNet"):
    if('model1' in locals()):
        del model1
        gc.collect()
    
    model1 = tf.keras.models.Sequential()

    model1.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, padding='same', input_shape=X_train.shape[1:]))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Conv2D(nb_filters, kernel_size))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model1.add(tf.keras.layers.Conv2D(2*nb_filters, kernel_size, padding='same'))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Conv2D(2*nb_filters, kernel_size))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model1.add(Conv2DNoise(int(nb_filters/2), kernel_size, padding='same', sigmanoise=sigmanoiseC))
    model1.add(tf.keras.layers.Activation('relu'))

    model1.add(tf.keras.layers.Flatten())
    model1.add(DenseNoise(1024,sigmanoise=sigmanoiseD))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Dense(nb_classes, name="last_layer"))
    model1.add(tf.keras.layers.Activation('softmax'))

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model1.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])