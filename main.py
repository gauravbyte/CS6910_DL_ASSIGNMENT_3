# %%
import random

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing on " + ("cuda:0" if torch.cuda.is_available() else "cpu"))
# %%
df_train = pd.read_csv("Data/hin/hin_train.csv")
df_test = pd.read_csv('Data/hin/hin_test.csv')
df_valid = pd.read_csv('Data/hin/hin_valid.csv')
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
    padded_input_batch = torch.nn.utils.rnn.pad_sequence(tensor_data, batch_first=True, padding_value=1).T
    padded_output_batch = torch.nn.utils.rnn.pad_sequence(tensor_target, batch_first=True, padding_value=1).T
    #     print(tensor_data)
    #     print(padded_input_batch.shape)
    #     print(padded_output_batch.shape)
    return padded_input_batch, padded_output_batch


# %%
train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, )

# %%
for data, target in train_dataloader:
    # print(data)
    # print(target)
    print(data.shape)
    print(target.shape)
    if True:
        break


# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        # print("encoder forward prop")

        embedded = self.dropout(self.embedding(src))
        # embedded = embedded.permute(1,0,2)
        # print("encoder embedded shape", embedded.shape)
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # print("encoder hidden dimension", hidden.shape)
        # print("encoder output dimension", outputs.shape)
        # print("encoder cell dimension", cell.shape)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell


# %%
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        # print("decoder hidden shape", hidden.shape)
        # print("decoder context shape", cell.shape)
        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # print("decoder embedded shape", embedded.shape)
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print("decoder output shape", output.shape)
        # print("decoder hidden shape", hidden.shape)
        # print("decoder context shape", cell.shape)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]
        # print("decoder prediction shape", prediction.shape)
        return prediction, hidden, cell


# %%
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
        # src = src.permute(1,0)
        # trg = trg.permute(1,0)
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        # print("seq2seq forward prop src shape", src.shape)
        # print("seq2seq forward prop trg shape", trg.shape)
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        # print("seq2seq trg vocab size", trg_vocab_size)
        # print("seq2seq batch size", batch_size)
        # print("seq2seq trg len", trg_len)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # print("seq2seq hidden shape", hidden.shape)
        # print("seq2seq cell shape", cell.shape)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # print("first input to decoder", input)
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# %%
INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
# %%
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
# %%
model = Seq2Seq(enc, dec, device).to(device)


# %%
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
# %%
optimizer = optim.Adam(model.parameters())
# %%
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss()


# %%
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, (data, target) in enumerate(iterator):
        # if(i == 2):
        #     break
        src = data
        trg = target

        optimizer.zero_grad()
        # print("train target shape", trg.shape)
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        # trg = trg.permute(1,0)
        # output = output.permute(1,0, 2)
        # print("train target shape", trg.shape)
        # print("train output shape", output.shape)
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        # print("target shape", trg.shape)
        # print("output shape", output.shape)
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #
        optimizer.step()
        #
        ls = loss.item()
        epoch_loss += ls
        print(ls)
    return epoch_loss / len(iterator)


# %% md
# N_EPOCHS = 10
# CLIP = 1
#
# best_valid_loss = float('inf')
#
# for epoch in range(N_EPOCHS):
#
#     # start_time = time.time()
#
#     train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
# valid_loss = evaluate(model, valid_iterator, criterion)
# %%
N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    # start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)

    # valid_loss = evaluate(model, valid_iterator, criterion)
