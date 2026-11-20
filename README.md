# Text Generation using LSTM

Word-level text generation model built with TensorFlow/Keras. Given a seed phrase, it predicts the next words one at a time.

## How it works
1. Tokenize text and build n-gram sequences
2. Pad sequences and split into inputs and target word
3. Train Embedding + LSTM + Softmax model (Adam, categorical cross-entropy)
4. Generate text word by word from a seed phrase

## Run
pip install tensorflow numpy ,
python text_generation.py

## Limitations
- Trained on a very small corpus, so output is repetitive and memorized
- Greedy (argmax) decoding

## Future improvements
- Larger dataset, dropout, validation split
- Temperature / top-k sampling
- GRU or Transformer comparison
