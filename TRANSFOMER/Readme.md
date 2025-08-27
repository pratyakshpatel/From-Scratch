# Transformer Architecture from Scratch

This is a minimal transformer implementation using only NumPy that demonstrates the core concepts behind modern language models like GPT. The architecture processes input tokens by converting them to embeddings, adding positional information, and passing them through transformer blocks that contain multi-head attention and feed-forward layers.

The multi-head attention mechanism is the key component that allows the model to focus on different parts of the input sequence simultaneously. It creates multiple attention heads that each learn different relationships in the data using scaled dot-product attention. The attention process computes scores between queries and keys, applies causal masking to prevent looking at future tokens, converts scores to probabilities with softmax, and uses these to weight the values.

Each transformer block combines attention with a feed-forward network that processes each position independently using two linear layers with ReLU activation. Both components use residual connections and layer normalization to help with training stability. The positional encoding uses sinusoidal functions to give the model information about token positions since it processes everything in parallel.

This simplified version removes dropout, complex initialization, and generation methods while keeping the essential transformer mechanisms. The model uses causal masking to ensure autoregressive behavior for text generation and outputs logits representing probability distributions over the vocabulary for each position.
