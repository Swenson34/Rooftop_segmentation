import tensorflow as tf

def Unet(img_width, img_height, img_channels):
    inputs_Unet = tf.keras.layers.Input((img_width, img_height, img_channels))

    # Scale the inputs between 0 and 1
    s = tf.keras.layers.Lambda(lambda x: x/255)(inputs_Unet)
    s = tf.cast(s, dtype = tf.float16) # Limit the precision

    # First convolution layer
    c1 = tf.keras.layers.Conv2D(16,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPool2D((2,2))(c1)

    # Second convolution layer
    c2 = tf.keras.layers.Conv2D(32,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPool2D((2,2))(c2)

    # Third convolution layer
    c3 = tf.keras.layers.Conv2D(64,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPool2D((2,2))(c3)

    # Fourth convolution layer
    c4 = tf.keras.layers.Conv2D(128,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPool2D((2,2))(c4)

    # Fifth convolution layer
    c5 = tf.keras.layers.Conv2D(256,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c5)

    # First transpose convolution layer
    u6 = tf.keras.layers.Conv2DTranspose(128,(3,3), strides = (2,2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c6)

    # Second transpose convolution layer
    u7 = tf.keras.layers.Conv2DTranspose(64,(3,3), strides = (2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c7)

    # Third transpose convolution layer
    u8 = tf.keras.layers.Conv2DTranspose(32,(3,3), strides = (2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c8)

    # Fourth transpose convolution layer
    u9 = tf.keras.layers.Conv2DTranspose(16,(3,3), strides = (2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16,(3,3), activation ='relu', kernel_initializer= 'he_normal', padding='same')(c9)

    outputs_Unet = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs_Unet], outputs = [outputs_Unet])

    return model


def FCN(img_width, img_height, img_channels):
    inputs_FCN = tf.keras.layers.Input((img_width, img_height, img_channels))

    c1 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs_FCN)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)

    p1 = tf.keras.layers.MaxPool2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)

    p2 = tf.keras.layers.MaxPool2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)

    p3 = tf.keras.layers.MaxPool2D((2, 2))(c4)

    dr1 = tf.keras.layers.Dropout(0.5)(p3)

    c5 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(dr1)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)

    d1 = tf.keras.layers.Dense(1024, activation='relu')(c5)
    d2 = tf.keras.layers.Dense(1024, activation='relu')(d1)

    # Deconvolution Layers (BatchNorm after non-linear activation)

    t1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same')(d2)
    t1 = tf.keras.layers.BatchNormalization()(t1)
    t1 = tf.keras.layers.Activation('relu')(t1)
    t1 = tf.keras.layers.UpSampling2D()(t1)

    t2 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same')(t1)
    t2 = tf.keras.layers.BatchNormalization()(t2)
    t2 = tf.keras.layers.Activation('relu')(t2)

    t3 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(t2)
    t3 = tf.keras.layers.BatchNormalization()(t3)
    t3 = tf.keras.layers.Activation('relu')(t3)
    t3 = tf.keras.layers.UpSampling2D()(t3)

    t4 = tf.keras.layers.Conv2DTranspose(16, (3, 3), padding='same')(t3)
    t4 = tf.keras.layers.BatchNormalization()(t4)
    t4 = tf.keras.layers.Activation('relu')(t4)
    t4 = tf.keras.layers.UpSampling2D()(t4)

    t5 = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same')(t4)
    t5 = tf.keras.layers.BatchNormalization()(t5)

    x = tf.keras.layers.Dropout(0.5)(t5)
    x = tf.keras.layers.Activation('sigmoid')(x)

    output_FCN = tf.keras.layers.Reshape((img_width, img_height, 1))(x)

    model = tf.keras.Model(inputs=inputs_FCN, outputs=output_FCN)

    return model


def FCN_deep(img_width, img_height, img_channels):
    # Encoding layer
    inputs_FCN = tf.keras.layers.Input((img_width, img_height, img_channels))

    c1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',strides= (1,1))(inputs_FCN)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(c1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c2 = tf.keras.layers.MaxPooling2D()(c2)
    c2 = tf.keras.layers.Dropout(0.25)(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(c2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)

    c4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(c3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c4 = tf.keras.layers.MaxPooling2D()(c4)
    c4 = tf.keras.layers.Dropout(0.25)(c4)

    c5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(c4)
    c5 = tf.keras.layers.BatchNormalization(name='bn5')(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)

    c6 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(c5)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)

    c7 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(c6)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.MaxPooling2D()(c7)
    c7 = tf.keras.layers.Dropout(0.25)(c7)

    c8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c7)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)

    c9 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c8)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)

    c10 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c9)
    c10 = tf.keras.layers.BatchNormalization()(c10)
    c10 = tf.keras.layers.Activation('relu')(c10)
    c10 = tf.keras.layers.MaxPooling2D()(c10)
    c10 = tf.keras.layers.Dropout(0.25)(c10)

    c11 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c10)
    c11 = tf.keras.layers.BatchNormalization()(c11)
    c11 = tf.keras.layers.Activation('relu')(c11)

    c12 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c11)
    c12 = tf.keras.layers.BatchNormalization()(c12)
    c12 = tf.keras.layers.Activation('relu')(c12)

    c13 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(c12)
    c13 = tf.keras.layers.BatchNormalization()(c13)
    c13 = tf.keras.layers.Activation('relu')(c13)
    c13 = tf.keras.layers.MaxPooling2D()(c13)
    c13 = tf.keras.layers.Dropout(0.25)(c13)

    d1 = tf.keras.layers.Dense(1024, activation = 'relu')(c13)
    d2 = tf.keras.layers.Dense(1024, activation = 'relu')(d1)

    # Decoding Layer

    t1 = tf.keras.layers.UpSampling2D()(d2)
    t1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t1)
    t1 = tf.keras.layers.BatchNormalization()(t1)
    t1 = tf.keras.layers.Activation('relu')(t1)

    t2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t1)
    t2 = tf.keras.layers.BatchNormalization()(t2)
    t2 = tf.keras.layers.Activation('relu')(t2)

    t3 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t2)
    t3 = tf.keras.layers.BatchNormalization()(t3)
    t3 = tf.keras.layers.Activation('relu')(t3)
    t3 = tf.keras.layers.UpSampling2D()(t3)
    t3 = tf.keras.layers.Dropout(0.25)(t3)

    t4 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t3)
    t4 = tf.keras.layers.BatchNormalization()(t4)
    t4 = tf.keras.layers.Activation('relu')(t4)

    t5 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t4)
    t5 = tf.keras.layers.BatchNormalization()(t5)
    t5 = tf.keras.layers.Activation('relu')(t5)

    t6 = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same')(t5)
    t6 = tf.keras.layers.BatchNormalization()(t6)
    t6 = tf.keras.layers.Activation('relu')(t6)
    t6 = tf.keras.layers.Dropout(0.25)(t6)

    t7 = tf.keras.layers.UpSampling2D()(t6)
    t7 = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same')(t7)
    t7 = tf.keras.layers.BatchNormalization()(t7)
    t7 = tf.keras.layers.Activation('relu')(t7)

    t8 = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same')(t7)
    t8 = tf.keras.layers.BatchNormalization()(t8)
    t8 = tf.keras.layers.Activation('relu')(t8)

    t9 = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same')(t8)
    t9 = tf.keras.layers.BatchNormalization()(t9)
    t9 = tf.keras.layers.Activation('relu')(t9)
    t9 = tf.keras.layers.Dropout(0.25)(t9)


    t10 = tf.keras.layers.UpSampling2D()(t9)
    t10 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same')(t10)
    t10 = tf.keras.layers.BatchNormalization()(t10)
    t10 = tf.keras.layers.Activation('relu')(t10)

    t11 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same')(t10)
    t11 = tf.keras.layers.BatchNormalization()(t11)
    t11 = tf.keras.layers.Activation('relu')(t11)


    t12 = tf.keras.layers.UpSampling2D()(t11)
    t12 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same')(t12)
    t12 = tf.keras.layers.BatchNormalization()(t12)
    t12 = tf.keras.layers.Activation('relu')(t12)
    t12 = tf.keras.layers.Dropout(0.3)(t12)

    t13 = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same')(t12)
    t13 = tf.keras.layers.BatchNormalization()(t13)
    pred = tf.keras.layers.Activation('sigmoid')(t13)

    model = tf.keras.Model(inputs=inputs_FCN, outputs=pred)

    return model