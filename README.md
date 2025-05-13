# 🔡 Seq2Seq RNN Model for Character-Level Transliteration (Latin → Devanagari)

This project implements a customizable **sequence-to-sequence (seq2seq)** deep learning model to transliterate words from Latin script to Devanagari at the character level using RNN-based architectures.

---

## 🚀 Overview

The model translates Latin-script inputs (e.g., "namaste") into their Devanagari equivalents (e.g., "नमस्ते") using a character-level encoder-decoder architecture. It supports various configurations of RNNs including standard RNN, GRU, and LSTM cells.

---

## 🧠 Model Architecture

### Components:
- **Input Layer**: Character embeddings for Latin characters.
- **Encoder**: RNN-based module that processes the input sequence.
- **Decoder**: RNN-based module that generates the Devanagari output, character by character.
- **Output Layer**: Fully connected layer projecting to target character probabilities.

### Supported Customizations:
- Embedding size `m`
- Hidden state size `k`
- Number of layers in encoder/decoder
- RNN cell type: `RNN`, `GRU`, or `LSTM`

---

## ⚙️ Computational Analysis

### (a) Total Number of Computations
Let:
- `T` = length of input/output sequence  
- `m` = embedding size  
- `k` = hidden size  
- `V` = vocabulary size (same for input and output)

Assumptions:
- 1-layer encoder and decoder
- BPTT used during training

**Total operations per forward-backward pass**:
```
O((2T + 1)mk + 2Tk² + 2Tk + (2T + 1)Vm + (T + 1)Vk + (T + 1)V)
```

---

### (b) Total Number of Parameters

| Component            | Parameters       |
|----------------------|------------------|
| Input Embedding      | `V × m`          |
| Encoder (1-layer LSTM)| `4 × (k² + k × m)` |
| Decoder (1-layer LSTM)| `4 × (k² + k × k)` |
| Output Layer         | `V × k`          |

> For LSTM, each layer includes 4 weight matrices per gate (input, forget, cell, output). Adjust accordingly if using GRU or vanilla RNN.

**Total parameters (approx):**
```
Vm + 8k² + Vk
```

---

## 📊 Visualizations

> Example training logs and attention visualizations.

![Training Loss and Accuracy](https://github.com/user-attachments/assets/05a0297f-867f-4e38-b8da-d56f559827e0)
![Attention Heatmap](https://github.com/user-attachments/assets/4dd6c11d-7551-475b-855e-772e81399535)
![Sample Output 1](https://github.com/user-attachments/assets/55279210-c154-4798-8ea4-d0f47033aa33)
![Sample Output 2](https://github.com/user-attachments/assets/f9334ca0-44f0-453b-9722-6bff53a9bb06)
![Sample Output 3](https://github.com/user-attachments/assets/99e85f18-fef1-41af-8591-d6071e9db3d4)
![Sample Output 4](https://github.com/user-attachments/assets/c8bebbbc-7415-4189-98a7-08f130ad21ba)

---

## 📁 File Structure

```
.
├── data/                   # Preprocessed datasets
├── model/                  # Encoder, decoder, training script
├── utils/                  # Helper functions for vocab, batching, metrics
├── plots/                  # Training plots, attention maps
├── train.py                # Main training script
├── evaluate.py             # Evaluation logic
└── README.md               # Project documentation
```

---

## 🧪 Future Work

- Add attention mechanism (Bahdanau/Luong)
- Integrate beam search decoding
- Expand to multilingual transliteration

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and distribute it!

---

## 🙌 Acknowledgments

This project was developed as part of CS6910 - Deep Learning at IIT Madras. Inspired by the effectiveness of sequence-to-sequence learning for NLP tasks.

---

## ✨ Example

**Input**: `namaste`  
**Output**: `नमस्ते`
