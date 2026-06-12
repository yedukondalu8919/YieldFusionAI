import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, inputs, return_attention=False):
        scores = self.score_dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        if return_attention:
            return context, tf.squeeze(weights, axis=-1)
        return context

def build_yieldnet_st(input_shape, dropout=0.3, l2_weight=1e-4, learning_rate=1e-3, auxiliary=False, n_crops=3):
    reg = regularizers.l2(l2_weight)
    inp = layers.Input(shape=input_shape, name="input_tensor")
    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg))(inp)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    x = layers.TimeDistributed(layers.Flatten(), name="flattened_spatial_features")(x)
    h = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=dropout), name="bi_lstm")(x)
    context = TemporalAttention(name="phenology_attention")(h)
    z = layers.Dense(64, activation="relu", kernel_regularizer=reg)(context)
    z = layers.Dropout(dropout)(z)
    yield_out = layers.Dense(1, activation="linear", name="yield_output")(z)
    if auxiliary:
        crop_out = layers.Dense(n_crops, activation="softmax", name="crop_type_output")(context)
        model = Model(inp, [yield_out, crop_out], name="YieldNet_ST")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss={"yield_output":"mse", "crop_type_output":"sparse_categorical_crossentropy"},
                      loss_weights={"yield_output":1.0, "crop_type_output":0.15},
                      metrics={"yield_output":[tf.keras.metrics.MeanAbsoluteError(name="mae")]})
    else:
        model = Model(inp, yield_out, name="YieldNet_ST")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def build_attention_model(model):
    att_layer = model.get_layer("phenology_attention")
    bi = model.get_layer("bi_lstm").output
    context, weights = att_layer(bi, return_attention=True)
    return Model(model.input, weights, name="attention_extractor")
