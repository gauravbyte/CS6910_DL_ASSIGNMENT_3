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

![CleanShot-03-27 at 00 31 132025CS22M045 CS6910 - Assignment 3  Deep Learning CS6910 Assignment 3 report – Weights   BiasesGoogle Chrome](https://github.com/user-attachments/assets/05a0297f-867f-4e38-b8da-d56f559827e0)

---
![CleanShot-03-27 at 00 31 582025CS22M045 CS6910 - Assignment 3  Deep Learning CS6910 Assignment 3 report – Weights   BiasesGoogle Chrome](https://github.com/user-attachments/assets/4dd6c11d-7551-475b-855e-772e81399535)

![CleanShot-03-27 at 00 32 402025CS22M045 CS6910 - Assignment 3  Deep Learning CS6910 Assignment 3 report – Weights   BiasesGoogle Chrome](https://github.com/user-attachments/assets/55279210-c154-4798-8ea4-d0f47033aa33)


![f0037f1a](https://github.com/user-attachments/assets/f9334ca0-44f0-453b-9722-6bff53a9bb06)


![ebd9111c](https://github.com/user-attachments/assets/99e85f18-fef1-41af-8591-d6071e9db3d4)

![6467443e](https://github.com/user-attachments/assets/c8bebbbc-7415-4189-98a7-08f130ad21ba)



## License

MIT License
