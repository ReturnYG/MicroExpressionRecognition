import tensorflow as tf
from tensorflow.keras import layers, backend
from tensorflow import keras

# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.


def correct_pad(inputs, kernel_size):
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same")
    return conv_layer(x)


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1, )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(local_features, filters=projection_dim, kernel_size=1, strides=strides)

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(local_features)
    global_features = transformer_block(non_overlapping_patches, num_blocks, projection_dim)

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(global_features)

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides)
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(local_global_features, filters=projection_dim, strides=strides)

    return local_global_features


def create_mobilevit(num_classes=5):
    inputs = keras.Input((image_size, image_size, 1))

    # Initial conv-stem -> MV2 block.
    x = conv_block(inputs, filters=16)
    x = inverted_residual_block(x, expanded_channels=16 * expansion_factor, output_channels=16)

    # Downsampling with MV2 block.
    x = inverted_residual_block(x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2)
    x = inverted_residual_block(x, expanded_channels=24 * expansion_factor, output_channels=24)
    x = inverted_residual_block(x, expanded_channels=24 * expansion_factor, output_channels=24)

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2)
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2)
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2)
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)
