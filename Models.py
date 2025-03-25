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