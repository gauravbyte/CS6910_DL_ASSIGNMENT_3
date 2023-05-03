# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-02T16:56:10.835392Z","iopub.execute_input":"2023-05-02T16:56:10.835885Z","iopub.status.idle":"2023-05-02T16:56:10.842258Z","shell.execute_reply.started":"2023-05-02T16:56:10.835846Z","shell.execute_reply":"2023-05-02T16:56:10.840539Z"}}
#  [code] {"jupyter":{"outputs_hidden":false}}

import os

# %% [code] {"execution":{"iopub.status.busy":"2023-05-02T16:56:11.120387Z","iopub.execute_input":"2023-05-02T16:56:11.121845Z","iopub.status.idle":"2023-05-02T16:56:11.132246Z","shell.execute_reply.started":"2023-05-02T16:56:11.121779Z","shell.execute_reply":"2023-05-02T16:56:11.129843Z"}}
print(os.environ['PYTHONPATH'])

# %% [code] {"execution":{"iopub.status.busy":"2023-05-02T16:56:14.775075Z","iopub.execute_input":"2023-05-02T16:56:14.775537Z","iopub.status.idle":"2023-05-02T16:56:14.784274Z","shell.execute_reply.started":"2023-05-02T16:56:14.775497Z","shell.execute_reply":"2023-05-02T16:56:14.782306Z"}}

# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Import data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-02T12:55:25.029709Z","iopub.execute_input":"2023-05-02T12:55:25.029972Z","iopub.status.idle":"2023-05-02T12:55:25.036930Z","shell.execute_reply.started":"2023-05-02T12:55:25.029945Z","shell.execute_reply":"2023-05-02T12:55:25.035849Z"}}
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-02T12:55:25.038645Z","iopub.execute_input":"2023-05-02T12:55:25.039311Z","iopub.status.idle":"2023-05-02T12:55:25.053271Z","shell.execute_reply.started":"2023-05-02T12:55:25.039269Z","shell.execute_reply":"2023-05-02T12:55:25.051976Z"}}
# Instantiates the device to be used as GPU/CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing on " + ("cuda:0" if torch.cuda.is_available() else "cpu"))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-02T13:01:09.838864Z","iopub.execute_input":"2023-05-02T13:01:09.839357Z","iopub.status.idle":"2023-05-02T13:01:09.920038Z","shell.execute_reply.started":"2023-05-02T13:01:09.839319Z","shell.execute_reply":"2023-05-02T13:01:09.918970Z"}}
if 'kaggle' in os.environ['PYTHONPATH']:
    print('Running on Kaggle')
    df_train = pd.read_csv("/kaggle/input/aksharantar/aksharantar_sampled/mai/mai_train.csv")
    df_test = pd.read_csv('/kaggle/input/aksharantar/aksharantar_sampled/mai/mai_test.csv')
    df_valid = pd.read_csv('/kaggle/input/aksharantar/aksharantar_sampled/mar/mar_valid.csv')
else:
    print('Running on local')
    df_train = pd.read_csv("Data/hin/hin_train.csv")
    df_test = pd.read_csv('Data/hin/hin_test.csv')
    df_valid = pd.read_csv('Data/hin/hin_valid.csv')

print(df_train.shape)
print(df_test.shape)
print(df_valid.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-02T13:01:33.639444Z","iopub.execute_input":"2023-05-02T13:01:33.639944Z","iopub.status.idle":"2023-05-02T13:01:33.646746Z","shell.execute_reply.started":"2023-05-02T13:01:33.639901Z","shell.execute_reply":"2023-05-02T13:01:33.645612Z"},"jupyter":{"outputs_hidden":false}}
df_train = df_train.iloc[:200]
df_test = df_test.iloc[:20]
df_valid = df_valid.iloc[:20]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-02T13:01:35.766617Z","iopub.execute_input":"2023-05-02T13:01:35.767172Z","iopub.status.idle":"2023-05-02T13:01:35.816057Z","shell.execute_reply.started":"2023-05-02T13:01:35.767108Z","shell.execute_reply":"2023-05-02T13:01:35.814821Z"}}
# %%
PAD_CHAR = '_'
EOW_CHAR = '|'
SOW_CHAR = '$'
BATCH_SIZE = 3
# %%
eng_alphabets = [chr(alpha) for alpha in range(ord('a'), ord('z') + 1)]
# eng_alpha2index = {pad_char:0}
in_dict = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(eng_alphabets):
    in_dict[alpha] = index + 3
print(in_dict)

# %%
hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)
out_dict = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(hindi_alphabets):
    out_dict[alpha] = index + 3
# %%
print(out_dict)
print(len(out_dict))
# %%
if df_train.iloc[0][0][0] != SOW_CHAR:
    df_train = df_train.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)
    df_test = df_test.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)
    df_valid = df_valid.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)

# %%
df_train = df_train.set_axis(['X', 'Y'], axis=1)
df_valid = df_valid.set_axis(['X', 'Y'], axis=1)
df_test = df_test.set_axis(['X', 'Y'], axis=1)
# %%
print(df_train)
print(df_test)
print(df_valid)

# %%
max_input_length = max(df_train.iloc[:, 0].apply(lambda x: len(x)).max(),
                       df_test.iloc[:, 0].apply(lambda x: len(x)).max(),
                       df_valid.iloc[:, 0].apply(lambda x: len(x)).max())

max_output_length = max(df_train.iloc[:, 1].apply(lambda x: len(x)).max(),
                        df_test.iloc[:, 1].apply(lambda x: len(x)).max(),
                        df_valid.iloc[:, 1].apply(lambda x: len(x)).max())

print("max input length", max_input_length)
print("max output length", max_output_length)
MAX_LENGTH = max(max_input_length, max_output_length)
print("max_length", MAX_LENGTH)
# %%
input_vocab_size = len(in_dict)
output_vocab_size = len(out_dict)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()

# %%
input_vocab_size = len(in_dict)
output_vocab_size = len(out_dict)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()


# %% md

# %%
def inputToTensor(line):
    # print([in_dict[x] for x in line])
    tensor = torch.tensor(data=([in_dict[x] for x in line]), dtype=torch.long)
    return tensor


def charToTensor(char, dic=in_dict):
    tensor = torch.zeros(len(dic))
    tensor[dic[char]] = 1
    return tensor


def outToTensor(word):
    tensor = torch.tensor([out_dict[x] for x in word])
    return tensor


# %%
print(inputToTensor(train_list[0][0]))
# %%
print(train_list[1][1])

# %%
inputToTensor("$bindhya|")
# %%
print(in_dict)
# %%
print(inputToTensor("hello"))


# %%
def generate_batch(data_batch):
    #     print(data_batch)
    tensor_data = [inputToTensor(x[0].ljust(max_input_length, PAD_CHAR)) for x in data_batch]
    tensor_target = [outToTensor(x[1].ljust(max_output_length, PAD_CHAR)) for x in data_batch]
    padded_input_batch = (torch.nn.utils.rnn.pad_sequence(tensor_data, batch_first=True, padding_value=1).T).to(device)
    padded_output_batch = (torch.nn.utils.rnn.pad_sequence(tensor_target, batch_first=True, padding_value=1).T).to(
        device)
    #     print(tensor_data)
    #     print(padded_input_batch.shape)
    #     print(padded_output_batch.shape)
    return padded_input_batch, padded_output_batch


train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


for data, target in train_dataloader:
    # print(data)
    # print(target)
    print(data.shape)
    print(target.shape)
    if True:
        break


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='lstm'):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):  # LSTM returns a tuple (hidden_state, cell_state)
            cell = hidden[1]
        else:
            cell = None
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='lstm'):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs



INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
HID_DIM = 32
N_LAYERS = 2
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
RNN_TYPE = 'rnn'
# %%

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
# %%
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (data, target) in enumerate(iterator):
        # if i==3:
        #     break
        src = data
        trg = target
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        preds = output.argmax(dim=1)
        print("preds:", preds)
        print("trg:", trg)
        ls = loss.item()
        epoch_loss += ls
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(iterator):
            # if i==3:
            #     break
            src = data
            trg = target
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print("epoch", epoch,end=" ")
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    # print("train_loss", train_loss,end=" ")
    # valid_loss = evaluate(model, valid_dataloader, criterion)
    # print("valid_loss", valid_loss)