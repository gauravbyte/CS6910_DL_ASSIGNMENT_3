import os
from tqdm.auto import tqdm
from datetime import time
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

import random
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing on " + ("cuda:0" if torch.cuda.is_available() else "cpu"))

if 'PYTHONPATH' in os.environ:
    if 'kaggle' in os.environ['PYTHONPATH']:
        print('Running on Kaggle')
        df_train = pd.read_csv("/kaggle/input/aksharantar_sampled/aksharantar_sampled/mar/mar_train.csv")
        df_valid = pd.read_csv('/kaggle/input/aksharantar_sampled/aksharantar_sampled/mar/mar_valid.csv')
        df_test = pd.read_csv('/kaggle/input/aksharantar_sampled/aksharantar_sampled/mar/mar_test.csv')
    else:
        # %%
        print('Running on local')
        df_train = pd.read_csv("Data/mar/mar_train.csv")
        df_test = pd.read_csv('Data/mar/mar_test.csv')
        df_valid = pd.read_csv('Data/mar/mar_valid.csv')
else:
    print('Running on local')
    df_train = pd.read_csv("Data/mar/mar_train.csv")
    df_test = pd.read_csv('Data/mar/mar_test.csv')
    df_valid = pd.read_csv('Data/mar/mar_valid.csv')

print(df_train.shape)
print(df_test.shape)
print(df_valid.shape)

PAD_CHAR = '_'  # padding character
EOW_CHAR = '|'  # end of word character
SOW_CHAR = '$'  # start of word character
BATCH_SIZE = 32
ENGLISH_ALPHA = [chr(alpha) for alpha in range(ord('a'), ord('z') + 1)]
INPUT_CHAR_INDX = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(ENGLISH_ALPHA):
    INPUT_CHAR_INDX[alpha] = index + 3


INPUT_INDEX_CHAR = {v: k for k, v in INPUT_CHAR_INDX.items()}

df_train = df_train.set_axis(['X', 'Y'], axis=1)
df_valid = df_valid.set_axis(['X', 'Y'], axis=1)
df_test = df_test.set_axis(['X', 'Y'], axis=1)

print(INPUT_CHAR_INDX)

ouput_words = df_train['Y'].tolist() + df_test['Y'].tolist() + df_valid['Y'].tolist()
output_char_set = set()
for word in ouput_words:
    for char in word:
        output_char_set.add(char)

OUT_ALPHA = list(output_char_set)
# OUT_ALPHA = [chr(alpha) for alpha in range(2304, 2432)]
OUT_ALPHA_SIZE = len(OUT_ALPHA)
OUT_CHAR_INDEX = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(OUT_ALPHA):
    OUT_CHAR_INDEX[alpha] = index + 3
# %%



print("ouput char set ",output_char_set)
print("ouput char set size",len(output_char_set))

OUTPUT_INDEX_CHAR = {v: k for k, v in OUT_CHAR_INDEX.items()}

# print(OUT_CHAR_INDEX)
# print(len(OUT_CHAR_INDEX))

df_train = df_train.set_axis(['X', 'Y'], axis=1)
df_valid = df_valid.set_axis(['X', 'Y'], axis=1)
df_test = df_test.set_axis(['X', 'Y'], axis=1)
# %%
print(df_train)
print(df_test)
print(df_valid)

if 'PYTHONPATH' in os.environ:
    if 'kaggle' not in os.environ['PYTHONPATH']:
        df_train = df_train.iloc[:2000,:]
        df_test = df_test.iloc[:2000,:]
        df_valid = df_valid.iloc[:200,:]

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
input_vocab_size = len(INPUT_CHAR_INDX)
output_vocab_size = len(OUT_CHAR_INDEX)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()

# %%
input_vocab_size = len(INPUT_CHAR_INDX)
output_vocab_size = len(OUT_CHAR_INDEX)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()


# %% md
class Transliterate(Dataset):
    def __init__(self, df_data, in_dict, out_dict):
        super().__init__()
        self.df_data_word = df_data.copy()
        self.in_dict = in_dict
        self.out_dict = out_dict
        self.df_data = df_data.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)


    def __get_random_word__(self):
        idx = random.randint(0, len(self.df_data))
        input_word = self.df_data_word.iloc[idx][0]
        output_word = self.df_data_word.iloc[idx][1]
        return input_word, output_word

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        input_word = self.df_data.iloc[idx][0]
        output_word = self.df_data.iloc[idx][1]
        input_tensor = inputToTensor(input_word)
        output_tensor = outToTensor(output_word)
        return input_tensor, output_tensor

    def __getrandom__(self):
        idx = random.randint(0,len(self.data_list))
        input_word = self.df_data[idx][0]
        output_word = self.df_data[idx][1]
        input_tensor = inputToTensor(input_word)
        output_tensor = outToTensor(output_word)
        return input_tensor, output_tensor

    def preprocess(self, word):
        return SOW_CHAR + word + EOW_CHAR


train_data = Transliterate(df_train, INPUT_CHAR_INDX, OUT_CHAR_INDEX)
valid_data = Transliterate(df_valid, INPUT_CHAR_INDX, OUT_CHAR_INDEX)
test_data = Transliterate(df_test, INPUT_CHAR_INDX, OUT_CHAR_INDEX)
# %%
def inputToTensor(line):
    tensor = torch.tensor(data=([INPUT_CHAR_INDX[x] for x in line]), dtype=torch.long)
    return tensor


def charToTensor(char, dic=INPUT_CHAR_INDX):
    tensor = torch.zeros(len(dic))
    tensor[dic[char]] = 1
    return tensor


def outToTensor(word):
    tensor = torch.tensor([OUT_CHAR_INDEX[x] for x in word])
    return tensor


# %%
print(inputToTensor(train_list[0][0]))
# %%
print(train_list[1][1])

# %%
inputToTensor("$bindhya|")
# %%
print(INPUT_CHAR_INDX)
# %%
print(inputToTensor("hello"))

# %%
def generate_batch(data_batch):
    train_batch = [x[0] for x in data_batch]
    target_batch = [x[1] for x in data_batch]
    train_pad = torch.nn.utils.rnn.pad_sequence(train_batch, batch_first=True, padding_value=0)
    train_pad = train_pad[:, :MAX_LENGTH]
    train_pad = torch.nn.functional.pad(train_pad, (0, MAX_LENGTH - train_pad.size(1)), value=0)
    target_pad = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=0)
    target_pad = target_pad[:, :MAX_LENGTH]
    target_pad = torch.nn.functional.pad(target_pad, (0, MAX_LENGTH - target_pad.size(1)), value=0)
    padded_input_batch = train_pad.T.to(device)
    padded_output_batch = target_pad.T.to(device)
    return padded_input_batch, padded_output_batch


# %%
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


for data, target in train_dataloader:
    print(data.shape)
    print(target.shape)
    if True:
        break


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, cell_type, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size,hidden_size,num_layers,dropout=dropout,bidirectional=bidirectional)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=dropout,bidirectional=bidirectional)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,dropout=dropout,bidirectional=bidirectional)
        # self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
        else:
            outputs, hidden = self.rnn(embedded)
        if(self.bidirectional == True):
            hidden = hidden[self.num_layers - 1] + hidden[self.num_layers - 1]
            hidden = hidden.repeat(self.num_layers,1,1)
            if(self.cell_type == "LSTM"):
                cell = cell[self.num_layers - 1] + cell[self.num_layers - 1]
                cell = cell.repeat(self.num_layers,1,1)
                return hidden, cell 
        if(self.cell_type == "LSTM"):
            return hidden, cell
        else:
            return outputs,hidden
        
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, num_layers, dropout, cell_type='LSTM', bidirectional=False):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.cell_type = cell_type
        if cell_type == 'RNN':
            self.rnn = nn.RNN(emb_dim, hidden_size, num_layers, dropout=dropout)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers, dropout=dropout)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, dropout=dropout)
        else:
            raise ValueError("Invalid RNN type specified!")
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded,( hidden, cell))
            prediction = self.fc_out(output).squeeze(0)
            return prediction, hidden, cell
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output).squeeze(0)
            return prediction, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device,teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # encoder_hidden = self.encoder.initHidden()
        if self.encoder.cell_type == 'LSTM':
            encoder_hidden, encoder_cell = self.encoder(src)
            hidden = encoder_hidden
            cell = encoder_cell
        else:
            output, hidden = self.encoder(src)
        input_decoder = trg[0, :]
        for t in range(1, trg_len):
            if self.decoder.cell_type == 'LSTM':
                dec_output, hidden, cell = self.decoder(input_decoder, hidden, cell)
            else:
                dec_output, hidden = self.decoder(input_decoder, hidden)
            outputs[t] = dec_output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            input_decoder = trg[t] if teacher_force else top1
        return outputs


# function to convert tensor batch back to words
def convert_tensor_to_input_words(batch):
    words = []
    for i in range(batch.shape[1]):
        word = ""
        for j in range(batch.shape[0]):
            char = INPUT_INDEX_CHAR[batch[j][i].item()]
            if char == PAD_CHAR or char == SOW_CHAR or char == EOW_CHAR:
                continue
            else:
                word += char
        words.append(word)

    return words

def convert_tensor_to_target_words(batch):
    words = []
    for i in range(batch.shape[1]):
        word = ""
        for j in range(batch.shape[0]):
            char = OUTPUT_INDEX_CHAR[batch[j][i].item()]
            if char == PAD_CHAR or char == SOW_CHAR or char == EOW_CHAR:
                continue
            else:
                word += char
        words.append(word)
    return words

def get_accuracy(preds, target):

    preds = preds.argmax(dim=2)
    preds = preds[1:]
    target = target[1:]
    matches = torch.eq(preds, target)
    columns_matches = torch.sum(matches, dim=0)
    num_matching_columns = torch.sum(columns_matches == target.shape[0])
    acc = num_matching_columns / target.shape[1]
    return acc.item()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    pbar = tqdm(iterator, desc="Training",position=0, leave=True)
    for i, (data, target) in enumerate(pbar):
        src = data.to(device)
        trg = target.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output_reshaped = output[1:].reshape(-1, output_dim)
        trg_reshaped = trg[1:].reshape(-1)
        loss = criterion(output_reshaped, trg_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        acc = get_accuracy(output, trg)
        # # convert src to words
        # INPUT_WORD_TRAIN.extend(convert_tensor_to_input_words(src))
        #
        # # convert trg to words
        # TARGET_WORD_TRAIN.extend(convert_tensor_to_target_words(trg))
        #
        # # convert output to words
        # PREDICTED_WORD_TRAIN.extend(convert_tensor_to_target_words(output.argmax(dim=2)))

        epoch_acc += acc
        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (i + 1), train_acc=epoch_acc / (i + 1))
        if i ==3 :
            break
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def input_output_predicted(model, iterator):
    INPUT_WORDS =[]
    TARGET_WORDS=[]
    PREDICTED_WORDS=[]
    for i, (data, target) in enumerate(iterator):
        src = data.to(device)
        trg = target.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        # convert src to words
        INPUT_WORDS.extend(convert_tensor_to_input_words(src))
        # convert trg to words
        TARGET_WORDS.extend(convert_tensor_to_target_words(trg))
        # convert output to words
        PREDICTED_WORDS.extend(convert_tensor_to_target_words(output.argmax(dim=2)))

    return INPUT_WORDS,TARGET_WORDS,PREDICTED_WORDS


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # with torch.no_grad():
    for i, (data, target) in enumerate(iterator):
        src = data.to(device)
        trg = target.to(device)
        output = model(src, trg)
        output_dim = output.shape[-1]
        output_reshaped = output[1:].reshape(-1, output_dim)
        trg_reshaped = trg[1:].reshape(-1)
        loss = criterion(output_reshaped, trg_reshaped)
        acc = get_accuracy(output,trg)
        epoch_acc += acc
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



N_EPOCHS = 1
CLIP = 1
INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HIDDEN_SIZE = 256
num_layers = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TEACHER_FORCING = 0.5
BI_DIRECTION = False
CELL_TYPE = 'LSTM'
pred_src = "$bindya|"
pred_trg = '$बिन्द्या|'
BATCH_SIZE = 32

enc = Encoder(input_size=INPUT_DIM, embedding_size=ENC_EMB_DIM, hidden_size=HIDDEN_SIZE, num_layers=num_layers, dropout=ENC_DROPOUT, cell_type=CELL_TYPE, bidirectional=BI_DIRECTION)
dec = Decoder(output_dim=OUTPUT_DIM, emb_dim=DEC_EMB_DIM, hidden_size=HIDDEN_SIZE, num_layers=num_layers, dropout=DEC_DROPOUT, cell_type=CELL_TYPE, bidirectional=BI_DIRECTION)

model = Seq2Seq(enc, dec, device, teacher_forcing_ratio=TEACHER_FORCING).to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

best_valid_loss = float('inf')
gbar = tqdm(range(1, N_EPOCHS + 1),position=1,leave=True, desc='Epochs', total=N_EPOCHS)
for epoch in gbar:
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, criterion)
    gbar.set_postfix(train_loss=train_loss, train_acc=train_accuracy, val_loss=valid_loss, val_acc=valid_accuracy)

INPUT_TRAIN_WORDS,TARGET_TRAIN_WORDS,PREDICTED_TRAIN_WORDS = input_output_predicted(model, train_dataloader)
INPUT_WORD_VALID,TARGET_WORD_VALID,PREDICTED_WORD_VALID = input_output_predicted(model, valid_dataloader)
INPUT_WORD_TEST,TARGET_WORD_TEST,PREDICTED_WORD_TEST = input_output_predicted(model, test_dataloader)
print("INPUT_WORD_TRAIN",INPUT_TRAIN_WORDS)
print("TARGET_WORD_TRAIN",TARGET_TRAIN_WORDS)
print("PREDICTED_WORD_TRAIN",PREDICTED_TRAIN_WORDS)
print("INPUT_WORD_VALID",INPUT_WORD_VALID)
print("TARGET_WORD_VALID",TARGET_WORD_VALID)
print("PREDICTED_WORD_VALID",PREDICTED_WORD_VALID)
print("INPUT_WORD_TEST",INPUT_WORD_TEST)
print("TARGET_WORD_TEST",TARGET_WORD_TEST)
print("PREDICTED_WORD_TEST",PREDICTED_WORD_TEST)
# predict(model,pred_src,pred_trg)
# predict(model,"$बिन्द्या|","$bindya|")