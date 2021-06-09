import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Dropout, LayerNormalization, Input, Conv2D, Reshape, UpSampling2D
from tensorflow.keras.activations import gelu


# +
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
        
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads, 
            name="MultiHeadDotProductAttention_1"
        )
        
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


# -

def VIT(
    image_size, 
    patch_size, 
    num_layers, 
    num_classes, 
    hidden_size, 
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
        
        
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    
    y = tf.keras.layers.Dense(num_classes, name="head", activation="linear")(y)
    
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)
