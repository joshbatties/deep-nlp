import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        p, i = np.meshgrid(np.arange(max_length),
                          2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True

    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]

class TransformerEncoder:
    def __init__(self,
                 vocab_size,
                 num_layers=6,
                 num_heads=8,
                 d_model=512,
                 dff=2048,
                 maximum_position_encoding=10000,
                 dropout_rate=0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        self.encoder_layers = [
            self._build_encoder_layer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def _build_encoder_layer(self, d_model, num_heads, dff, dropout_rate):
        inputs = tf.keras.Input(shape=(None, d_model))
        
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )(inputs + attention)
        
        # Feed forward network
        outputs = tf.keras.layers.Dense(dff, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(d_model)(outputs)
        outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )(attention + outputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def build_model(self, sequence_length):
        inputs = tf.keras.Input(shape=(sequence_length,))
        
        # Embedding and positional encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            
        return tf.keras.Model(inputs=inputs, outputs=x)
    
class TransformerWithClassification(TransformerEncoder):
    def build_classifier(self, sequence_length, num_classes):
        base_model = self.build_model(sequence_length)
        x = tf.keras.layers.GlobalAveragePooling1D()(base_model.output)
        x = tf.keras.layers.Dense(self.d_model, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        return model
