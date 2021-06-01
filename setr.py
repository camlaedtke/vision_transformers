import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Dropout, LayerNormalization, Input, Conv2D, Reshape, UpSampling2D
from tensorflow.keras.activations import gelu


class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    
    

class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(shape=(1, input_shape[1], input_shape[2])),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = Dense(hidden_size, name="query")
        self.key_dense = Dense(hidden_size, name="key")
        self.value_dense = Dense(hidden_size, name="value")
        self.combine_heads = Dense(hidden_size, name="out")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights
    
    
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(num_heads=self.num_heads, name="MultiHeadDotProductAttention_1")
        self.mlpblock = tf.keras.Sequential(
            [
                Dense(self.mlp_dim, activation="linear", name=f"{self.name}/Dense_0"),
                Lambda(lambda x: gelu(x, approximate=False)),
                Dropout(self.dropout),
                Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        self.dropout_layer = Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        return {
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        }
    
    
def aux_head(x, n, image_size, hidden_size, num_classes):
    y_aux = Lambda(lambda v: v[:, 1:], name="Aux_{}_ExtractToken".format(n))(x)
    y_aux = LayerNormalization(epsilon=1e-6, name="Aux_{}_norm".format(n))(y_aux)
    y_aux = Reshape(target_shape=(int(image_size//16), int(image_size//16), hidden_size))(y_aux)
    
    y_aux = Conv2D(hidden_size, kernel_size=(3,3), strides=(1,1), padding='same', 
                   name="Aux_{}_conv_1".format(n))(y_aux)
    y_aux = BatchNormalization(name="Aux_{}_bn".format(n))(y_aux)
    y_aux = Activation("relu", name="Aux_{}_relu".format(n))(y_aux)
    y_aux = UpSampling2D(size=(8,8), interpolation='bilinear', name="Aux_{}_upsample_1".format(n))(y_aux)
    y_aux = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same', name="Aux_{}_conv_2".format(n))(y_aux)
    y_aux = UpSampling2D(size=(8,8), interpolation='bilinear', name="Aux_{}_upsample_2".format(n))(y_aux)
    
    y_aux = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding='same', 
                     dtype='float32', name="Aux_{}_output".format(n))(y_aux)
    
    return y_aux


def get_aux_heads(y_list, image_size, hidden_size, num_classes, aux_layers):
    
    y_aux_list = []
    for i in range(0, len(y_list)):
        y_aux = aux_head(y_list[i], aux_layers[i], image_size, hidden_size, num_classes)
        y_aux_list.append(y_aux)
        
    return list(y_aux_list)


def get_decode_head(y, image_size, hidden_size, num_classes):
    
    y = Reshape(target_shape=(int(image_size//16), int(image_size//16), hidden_size))(y)
    
    y = UpSampling2D(size=(2,2), interpolation='bilinear', name="Decode_upsample_1")(y)
    y = Conv2D(hidden_size, kernel_size=(3,3), strides=(1,1), padding='same', name="Decode_conv_1")(y)
    y = BatchNormalization(name="Decode_bn_1")(y)
    y = Activation("relu", name="Decode_relu_1")(y)

    y = UpSampling2D(size=(2,2), interpolation='bilinear', name="Decode_upsample_2")(y)
    y = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', name="Decode_conv_2")(y)
    y = BatchNormalization(name="Decode_bn_2")(y)
    y = Activation("relu", name="Decode_relu_2")(y)

    y = UpSampling2D(size=(2,2), interpolation='bilinear', name="Decode_upsample_3")(y)
    y = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', name="Decode_conv_3")(y)
    y = BatchNormalization(name="Decode_bn_3")(y)
    y = Activation("relu", name="Decode_relu_3")(y)

    y = UpSampling2D(size=(2,2), interpolation='bilinear', name="Decode_upsample_4")(y)
    y = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', name="Decode_conv_4")(y)
    y = BatchNormalization(name="Decode_bn_4")(y)
    y = Activation("relu", name="Decode_relu_4")(y)
    
    y = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding='same', 
        dtype='float32', name="Decode_output")(y)
    
    return y

    
def SETR_PUP(
    image_size, 
    patch_size, 
    num_layers, 
    num_classes, 
    hidden_size, 
    aux_layers,
    num_heads, 
    name, 
    mlp_dim, 
    dropout=0.1
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
    """
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    
    x = Input(shape=(image_size, image_size, 3))
    y = Conv2D(
        filters=hidden_size, 
        kernel_size=patch_size, 
        strides=patch_size, 
        padding="valid", 
        name="embedding"
    )(x)
    y = Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = ClassToken(name="class_token")(y)
    y = AddPositionEmbs(name="Transformer/posembed_input")(y)
    
    y_list = []
    for n in range(num_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            name=f"Transformer/encoderblock_{n}"
        )(y)
        
        if aux_layers is not None:
            if n in aux_layers:
                y_list.append(y)
            
    if aux_layers is not None:
        y_aux_list = get_aux_heads(y_list, image_size, hidden_size, num_classes, aux_layers)
            
    y = Lambda(lambda v: v[:, 1:], name="ExtractToken")(y)
    y = LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm_decode")(y)
    
    y = get_decode_head(y, image_size, hidden_size, num_classes)

    if aux_layers is not None:
        out_heads = [y] + y_aux_list
    else:
        out_heads = [y]
    
    return tf.keras.models.Model(inputs=x, outputs=out_heads, name=name)