import tensorflow as tf

class EncoderDecoder:
    def __init__(self, 
                 input_vocab_size,
                 output_vocab_size, 
                 embedding_dim=256,
                 units=512):
        self.embedding_dim = embedding_dim
        self.units = units
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(
                self.input_vocab_size, 
                self.embedding_dim,
                mask_zero=True
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.units,
                    return_sequences=True,
                    return_state=True
                )
            )
        ])
    
    def _build_decoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(
                self.output_vocab_size,
                self.embedding_dim,
                mask_zero=True
            ),
            tf.keras.layers.LSTM(
                self.units * 2,  # *2 because encoder is bidirectional
                return_sequences=True
            ),
            tf.keras.layers.Dense(self.output_vocab_size, activation='softmax')
        ])
    
    def create_model(self):
        # Model inputs
        encoder_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
        decoder_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
        
        # Get encoder outputs and states
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder(encoder_inputs)
        
        # Concatenate states for decoder
        state_h = tf.concat([forward_h, backward_h], axis=-1)
        state_c = tf.concat([forward_c, backward_c], axis=-1)
        encoder_states = [state_h, state_c]
        
        # Create attention layer
        attention = tf.keras.layers.Attention()
        
        # Process decoder inputs
        decoder_outputs = self.decoder(decoder_inputs)
        
        # Apply attention
        attention_output = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate attention output with decoder output
        decoder_concat = tf.keras.layers.Concatenate()(
            [decoder_outputs, attention_output]
        )
        
        # Final dense layer
        outputs = tf.keras.layers.Dense(
            self.output_vocab_size, 
            activation='softmax'
        )(decoder_concat)
        
        # Create model
        model = tf.keras.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=outputs
        )
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
