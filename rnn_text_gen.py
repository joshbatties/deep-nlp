import tensorflow as tf
import numpy as np

class CharRNNGenerator:
    def __init__(self, vocab_size, embedding_size=16, rnn_units=128):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_units = rnn_units
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size
            ),
            tf.keras.layers.GRU(
                self.rnn_units,
                return_sequences=True
            ),
            tf.keras.layers.Dense(
                self.vocab_size,
                activation="softmax"
            )
        ])
        return model
    
    def compile(self, learning_rate=0.001):
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )
        
    def train(self, train_dataset, validation_dataset=None, epochs=10):
        return self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs
        )
        
    def generate_text(self, start_text, temperature=1.0, num_chars=1000):
        text = start_text
        for _ in range(num_chars):
            y_proba = self.model.predict([text])[0, -1:]
            rescaled_logits = tf.math.log(y_proba) / temperature
            char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
            text += self.vocab_to_char(char_id)
        return text
    
    def vocab_to_char(self, char_id):
        """Override this method to implement character conversion"""
        raise NotImplementedError
