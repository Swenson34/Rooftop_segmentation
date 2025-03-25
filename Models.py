def Unet():
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


def FCN():
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


def FCN_deep_model():
    # Encoding layer
    inputs_SegNet = tf.keras.layers.Input((img_width, img_height, img_channels))
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv1', strides=(1, 1))(inputs_SegNet)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv4')(x)
    x = tf.keras.layers.BatchNormalization(name='bn4')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv5')(x)
    x = tf.keras.layers.BatchNormalization(name='bn5')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv6')(x)
    x = tf.keras.layers.BatchNormalization(name='bn6')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv7')(x)
    x = tf.keras.layers.BatchNormalization(name='bn7')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv8')(x)
    x = tf.keras.layers.BatchNormalization(name='bn8')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv9')(x)
    x = tf.keras.layers.BatchNormalization(name='bn9')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv10')(x)
    x = tf.keras.layers.BatchNormalization(name='bn10')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv11')(x)
    x = tf.keras.layers.BatchNormalization(name='bn11')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv12')(x)
    x = tf.keras.layers.BatchNormalization(name='bn12')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv13')(x)
    x = tf.keras.layers.BatchNormalization(name='bn13')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(1024, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(x)

    # Decoding Layer

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn14')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn15')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn16')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv4')(x)
    x = tf.keras.layers.BatchNormalization(name='bn17')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv5')(x)
    x = tf.keras.layers.BatchNormalization(name='bn18')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
    x = tf.keras.layers.BatchNormalization(name='bn19')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', name='deconv7')(x)
    x = tf.keras.layers.BatchNormalization(name='bn20')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', name='deconv8')(x)
    x = tf.keras.layers.BatchNormalization(name='bn21')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
    x = tf.keras.layers.BatchNormalization(name='bn22')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', name='deconv10')(x)
    x = tf.keras.layers.BatchNormalization(name='bn23')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
    x = tf.keras.layers.BatchNormalization(name='bn24')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', name='deconv12')(x)
    x = tf.keras.layers.BatchNormalization(name='bn25')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
    x = tf.keras.layers.BatchNormalization(name='bn26')(x)
    pred = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=inputs_SegNet, outputs=pred)

    return model