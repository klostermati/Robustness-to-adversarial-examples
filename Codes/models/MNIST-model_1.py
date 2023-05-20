# Training parameters
batch_size = 128
nb_epoch = 20
nb_filters = 32
pool_size = (3, 3)
kernel_size = (3, 3)

# Model
training = False
if(not ('model' in locals())):
    training = True
    print("Creating model")

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, padding="valid", input_shape=input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(nb_filters, kernel_size))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(tf.keras.layers.Conv2D(16, kernel_size, padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(nb_classes))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, validation_data=(X_test, Y_test))
    

# Model 1
if((not training) and args.detection == "stoGauNet"):
    if('model1' in locals()):
        del model1
        gc.collect()
    model1 = tf.keras.Sequential()

    model1.add(tf.keras.layers.Conv2D(nb_filters, kernel_size, padding="valid", input_shape=input_shape))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Conv2D(nb_filters, kernel_size))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    model1.add(Conv2DNoise(16, kernel_size, padding='valid',sigmanoise=sigmanoiseC))
    model1.add(tf.keras.layers.Activation('relu'))

    model1.add(tf.keras.layers.Flatten())
    model1.add(DenseNoise(128,sigmanoise=sigmanoiseD))
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Dense(nb_classes, name="last_layer"))
    model1.add(tf.keras.layers.Activation('softmax'))

    model1.compile( loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])