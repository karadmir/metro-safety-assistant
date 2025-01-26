import tensorflow as tf


def upsample(filters, size) -> tf.keras.Sequential:

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result


def unet_model() -> tf.keras.Model:
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False
    )

    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    backbone_outputs = [backbone.get_layer(name).output for name in layer_names]

    encoder = tf.keras.Model(inputs=backbone.input, outputs=backbone_outputs)
    encoder.trainable = False

    decoder = [
        upsample(512, 3), # 4x4 -> 8x8
        upsample(256, 3), # 8x8 -> 16x16
        upsample(128, 3), # 16x16 -> 32x32
        upsample(64, 3), # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    skips = encoder(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(decoder, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding='same', activation='sigmoid'
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
