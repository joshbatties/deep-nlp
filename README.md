# Deep Learning for NLP

This repository contains implementations of various Natural Language Processing (NLP) models using TensorFlow/Keras, including RNNs, Encoder-Decoder architectures, and Transformers.

## Features

- Character-level RNN text generation
- Sequence-to-sequence models with attention
- Transformer architecture implementation
- Support for:
  - Text generation (Shakespeare-style text)
  - Machine translation
  - Sentiment analysis
  - Date format conversion

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-nlp.git
cd deep-nlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Text Generation

```python
from models.rnn_text_gen import CharRNNGenerator

# Initialize and train the model
model = CharRNNGenerator(vocab_size=39)  # for Shakespeare text
model.compile()
model.train(train_dataset, validation_dataset)

# Generate text
generated_text = model.generate_text(
    start_text="To be or not to be",
    temperature=0.7
)
```

### Sequence-to-Sequence Translation

```python
from models.seq2seq import EncoderDecoder

# Create model
model = EncoderDecoder(
    input_vocab_size=1000,
    output_vocab_size=1000
)

# Build and compile
seq2seq = model.create_model()
model.compile_model(seq2seq)

# Train
seq2seq.fit(
    [encoder_input, decoder_input],
    decoder_output,
    epochs=10
)
```

### Transformer Model

```python
from models.transformer import TransformerWithClassification

# Create model for classification
transformer = TransformerWithClassification(
    vocab_size=10000,
    num_layers=6,
    num_heads=8
)

# Build classifier
model = transformer.build_classifier(
    sequence_length=512,
    num_classes=2
)

# Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)
```

## Models

### Character RNN
- Uses GRU cells for text generation
- Supports temperature-based sampling
- Includes embedding layer for character representations

### Encoder-Decoder
- Bidirectional LSTM encoder
- Attention mechanism
- Support for variable length sequences
- Teacher forcing during training

### Transformer
- Multi-head attention
- Positional encoding
- Layer normalization
- Support for classification tasks


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Based on techniques from:
- "Attention is All You Need" https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf (my absolute favourite paper ever XD)
- TensorFlow tutorials and documentation
- Various other deep learning for NLP resources
