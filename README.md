# Seq2Seq RNN Model for Character-Level Transliteration (Latin → Devanagari)

This project implements a flexible RNN-based sequence-to-sequence (seq2seq) model for character-level transliteration from Latin characters to Devanagari script.

## Architecture

The model contains the following components:

1. **Input Layer**: Character embedding layer.
2. **Encoder**: An RNN (can be RNN, LSTM, or GRU) that encodes the input sequence.
3. **Decoder**: Another RNN that decodes the representation into the target sequence, one character at a time, using the final hidden state of the encoder as input.

The model supports customization of:
- Embedding size (`m`)
- Hidden state size (`k`)
- Number of layers in encoder and decoder
- Type of recurrent cell (`RNN`, `LSTM`, or `GRU`)

---

## (a) Total Number of Computations

Let:
- `T` = length of input/output sequence  
- `m` = input embedding size  
- `k` = hidden state size  
- `V` = vocabulary size (same for source and target languages)

Assuming:
- 1 layer for encoder and decoder
- Same sequence length `T` for both encoder and decoder
- BPTT (Backpropagation Through Time)

**Total number of computations**:  
```
O((2T + 1)mk + 2Tk² + 2Tk + (2T + 1)Vm + (T + 1)Vk + (T + 1)V)
```

---

## (b) Total Number of Parameters

Breakdown:
- **Input Embedding Layer**: `Vm` parameters
- **Encoder RNN (1 layer)**: `4k²` parameters
- **Decoder RNN (1 layer)**: `4k²` parameters
- **Output Layer**: `Vk` parameters

> (Note: For LSTM, each layer has 4 sets of weights per gate. Adjust if using plain RNN or GRU.)

**Total number of parameters**:
```
Vm + 8k² + Vk
```

---

## License

MIT License
