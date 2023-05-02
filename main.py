# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:28.403038Z","iopub.execute_input":"2023-04-27T06:42:28.404141Z","iopub.status.idle":"2023-04-27T06:42:28.429613Z","shell.execute_reply.started":"2023-04-27T06:42:28.404094Z","shell.execute_reply":"2023-04-27T06:42:28.428560Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Import data


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:28.915876Z","iopub.execute_input":"2023-04-27T06:42:28.916417Z","iopub.status.idle":"2023-04-27T06:42:28.924580Z","shell.execute_reply.started":"2023-04-27T06:42:28.916370Z","shell.execute_reply":"2023-04-27T06:42:28.923557Z"}}
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:29.142889Z","iopub.execute_input":"2023-04-27T06:42:29.143724Z","iopub.status.idle":"2023-04-27T06:42:29.149644Z","shell.execute_reply.started":"2023-04-27T06:42:29.143682Z","shell.execute_reply":"2023-04-27T06:42:29.148744Z"}}
# Instantiates the device to be used as GPU/CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Executing on " + ("cuda:0" if torch.cuda.is_available() else "cpu"))



df_train = pd.read_csv("Data/hin/hin_train.csv")
df_test = pd.read_csv('Data/hin/hin_test.csv')
df_valid = pd.read_csv('Data/hin/hin_valid.csv')
# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:28.403038Z","iopub.execute_input":"2023-04-27T06:42:28.404141Z","iopub.status.idle":"2023-04-27T06:42:28.429613Z","shell.execute_reply.started":"2023-04-27T06:42:28.404094Z","shell.execute_reply":"2023-04-27T06:42:28.428560Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas




# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:53.474019Z","iopub.execute_input":"2023-04-27T06:55:53.474452Z","iopub.status.idle":"2023-04-27T06:55:53.480573Z","shell.execute_reply.started":"2023-04-27T06:55:53.474414Z","shell.execute_reply":"2023-04-27T06:55:53.479185Z"}}
PAD_CHAR = '_'
EOW_CHAR = '|'
SOW_CHAR = '$'
BATCH_SIZE = 1

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:53.627313Z","iopub.execute_input":"2023-04-27T06:55:53.628366Z","iopub.status.idle":"2023-04-27T06:55:53.636112Z","shell.execute_reply.started":"2023-04-27T06:55:53.628309Z","shell.execute_reply":"2023-04-27T06:55:53.634809Z"}}
eng_alphabets = [chr(alpha) for alpha in range(ord('a'), ord('z') + 1)]
# eng_alpha2index = {pad_char:0}
in_dict = {PAD_CHAR: 2, EOW_CHAR: 1, SOW_CHAR: 0}
for index, alpha in enumerate(eng_alphabets):
	in_dict[alpha] = index + 3
print(in_dict)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:53.799809Z","iopub.execute_input":"2023-04-27T06:55:53.801002Z","iopub.status.idle":"2023-04-27T06:55:53.808572Z","shell.execute_reply.started":"2023-04-27T06:55:53.800960Z","shell.execute_reply":"2023-04-27T06:55:53.807283Z"}}
hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)
out_dict = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(hindi_alphabets):
	out_dict[alpha] = index + 3

print(out_dict)
print(len(out_dict))

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:53.945211Z","iopub.execute_input":"2023-04-27T06:55:53.946425Z","iopub.status.idle":"2023-04-27T06:55:53.957754Z","shell.execute_reply.started":"2023-04-27T06:55:53.946380Z","shell.execute_reply":"2023-04-27T06:55:53.956154Z"}}
df_train = df_train.set_axis(['X', 'Y'], axis=1)
df_valid = df_valid.set_axis(['X', 'Y'], axis=1)
df_test = df_test.set_axis(['X', 'Y'], axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.089095Z","iopub.execute_input":"2023-04-27T06:55:54.089525Z","iopub.status.idle":"2023-04-27T06:55:54.106860Z","shell.execute_reply.started":"2023-04-27T06:55:54.089486Z","shell.execute_reply":"2023-04-27T06:55:54.105205Z"}}
# df_train= df_train.rename(columns={0: 'data', 1: 'target'})

if df_train.iloc[0][0][0] != SOW_CHAR:
	df_train = df_train.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)
	df_test = df_test.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)
	df_valid = df_valid.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)

# add_sow_eow(df_train,df_test,df_valid)
print(df_train)
print(df_test)
print(df_valid)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.246536Z","iopub.execute_input":"2023-04-27T06:55:54.247489Z","iopub.status.idle":"2023-04-27T06:55:54.311584Z","shell.execute_reply.started":"2023-04-27T06:55:54.247444Z","shell.execute_reply":"2023-04-27T06:55:54.310415Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.393819Z","iopub.execute_input":"2023-04-27T06:55:54.394247Z","iopub.status.idle":"2023-04-27T06:55:54.417700Z","shell.execute_reply.started":"2023-04-27T06:55:54.394183Z","shell.execute_reply":"2023-04-27T06:55:54.416487Z"}}
input_vocab_size = len(in_dict)
output_vocab_size = len(out_dict)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train = df_train.values.tolist()
valid = df_valid.values.tolist()
test = df_test.values.tolist()


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.541071Z","iopub.execute_input":"2023-04-27T06:55:54.541859Z","iopub.status.idle":"2023-04-27T06:55:54.549480Z","shell.execute_reply.started":"2023-04-27T06:55:54.541813Z","shell.execute_reply":"2023-04-27T06:55:54.548301Z"}}
# working
# def inputToTensor(line,diction=in_dict,max_size=-1):
#     tensor = torch.zeros(max(len(line),MAX_LENGTH), 1, input_vocab_size)
#     for li, letter in enumerate(line):
#         tensor[li][0][diction[letter]] = 1
#     return tensor

# def inputToTensor(line, diction=in_dict, max_size=-1):
# 	tensor = torch.zeros(max(len(line), MAX_LENGTH), input_vocab_size)
# 	for li, letter in enumerate(line):
# 		tensor[li][diction[letter]] = 1
# 	return tensor
def inputToTensor(line , diction=in_dict, max_size=-1):
	tensor = torch.tensor(in_dict[x] for x in line)

def charToTensor(char, dic=in_dict):
	tensor = torch.zeros(len(dic))
	tensor[dic[char]] = 1
	return tensor


def outToTensor(word):
	tensor = torch.tensor([out_dict[x] for x in word])
	return tensor


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.854345Z","iopub.execute_input":"2023-04-27T06:55:54.855254Z","iopub.status.idle":"2023-04-27T06:55:54.869648Z","shell.execute_reply.started":"2023-04-27T06:55:54.855181Z","shell.execute_reply":"2023-04-27T06:55:54.868472Z"}}
print(outToTensor("फ्रेंडलिस्टीत"))
inputToTensor(train[0][0], in_dict)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:54.993188Z","iopub.execute_input":"2023-04-27T06:55:54.993616Z","iopub.status.idle":"2023-04-27T06:55:55.001439Z","shell.execute_reply.started":"2023-04-27T06:55:54.993579Z","shell.execute_reply":"2023-04-27T06:55:55.000071Z"}}
def generate_batch(data_batch):
	#     print(data_batch)
	tensor_data = [inputToTensor(x[0].ljust(max_input_length, PAD_CHAR)) for x in data_batch]
	tensor_target = [outToTensor(x[1].ljust(max_output_length, PAD_CHAR)) for x in data_batch]
	padded_input_batch = torch.nn.utils.rnn.pad_sequence(tensor_data, batch_first=True, padding_value=1)
	padded_output_batch = torch.nn.utils.rnn.pad_sequence(tensor_target, batch_first=True, padding_value=1)
	#     print(tensor_data)
	#     print(padded_input_batch.shape)
	#     print(padded_output_batch.shape)
	return padded_input_batch, padded_output_batch


#     print(tensor_target)
#     target_batch = [torch.tensor(out_dict[x[1]]) for x in data_batch]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:55:55.165376Z","iopub.execute_input":"2023-04-27T06:55:55.166226Z","iopub.status.idle":"2023-04-27T06:55:55.172332Z","shell.execute_reply.started":"2023-04-27T06:55:55.166175Z","shell.execute_reply":"2023-04-27T06:55:55.171005Z"}}
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, )


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:16.920179Z","iopub.execute_input":"2023-04-27T07:05:16.920604Z","iopub.status.idle":"2023-04-27T07:05:16.929815Z","shell.execute_reply.started":"2023-04-27T07:05:16.920571Z","shell.execute_reply":"2023-04-27T07:05:16.928519Z"}}
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
		#         embedded = self.dropout(self.embedding(src))
		print("encoder forward prop")
		src = src.long()
		#         src = src.view(-1,BATCH_SIZE,max_input_length)
		embedded = self.embedding(src)
		embedded = embedded.reshape(-1, BATCH_SIZE, ENC_EMB_DIM)
		# embedded = [src len, batch size, emb dim]
		print("src shape", embedded.shape)
		outputs, (hidden, cell) = self.rnn(embedded)
		print("after rnn in encoder", src.shape)
		# outputs = [src len, batch size, hid dim * n directions]
		# hidden = [n layers * n directions, batch size, hid dim]
		# cell = [n layers * n directions, batch size, hid dim]
		# outputs are always from the top hidden layer
		return hidden, cell


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:17.187838Z","iopub.execute_input":"2023-04-27T07:05:17.188523Z","iopub.status.idle":"2023-04-27T07:05:17.198733Z","shell.execute_reply.started":"2023-04-27T07:05:17.188486Z","shell.execute_reply":"2023-04-27T07:05:17.197733Z"}}
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

		input = input.unsqueeze(0)
		print("input dimensions", input.dim)
		# input = [1, batch size]

		#         embedded = self.dropout(self.embedding(input))
		embedded = self.embedding(input)

		# embedded = [1, batch size, emb dim]

		output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

		# output = [seq len, batch size, hid dim * n directions]
		# hidden = [n layers * n directions, batch size, hid dim]
		# cell = [n layers * n directions, batch size, hid dim]

		# seq len and n directions will always be 1 in the decoder, therefore:
		# output = [1, batch size, hid dim]
		# hidden = [n layers, batch size, hid dim]
		# cell = [n layers, batch size, hid dim]

		prediction = self.fc_out(output.squeeze(0))

		# prediction = [batch size, output dim]

		return prediction, hidden, cell



# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:17.483756Z","iopub.execute_input":"2023-04-27T07:05:17.484150Z","iopub.status.idle":"2023-04-27T07:05:17.496016Z","shell.execute_reply.started":"2023-04-27T07:05:17.484115Z","shell.execute_reply":"2023-04-27T07:05:17.494664Z"}}
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
		# src = [src len, batch size]
		# trg = [trg len, batch size]
		# teacher_forcing_ratio is probability to use teacher forcing
		# e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

		batch_size = trg.shape[0]
		print("batch_size ", batch_size)
		trg_len = trg.shape[1]
		print("trg_len ", trg_len)
		trg_vocab_size = self.decoder.output_dim

		# tensor to store decoder outputs
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

		# last hidden state of the encoder is used as the initial hidden state of the decoder
		hidden, cell = self.encoder(src)

		# first input to the decoder is the <sos> tokens
		input = trg[0, :]

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


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:17.656410Z","iopub.execute_input":"2023-04-27T07:05:17.657110Z","iopub.status.idle":"2023-04-27T07:05:17.751172Z","shell.execute_reply.started":"2023-04-27T07:05:17.657067Z","shell.execute_reply":"2023-04-27T07:05:17.750232Z"}}
INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:17.753417Z","iopub.execute_input":"2023-04-27T07:05:17.754205Z","iopub.status.idle":"2023-04-27T07:05:17.954453Z","shell.execute_reply.started":"2023-04-27T07:05:17.754155Z","shell.execute_reply":"2023-04-27T07:05:17.953269Z"}}
def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:17.956634Z","iopub.execute_input":"2023-04-27T07:05:17.957408Z","iopub.status.idle":"2023-04-27T07:05:17.965796Z","shell.execute_reply.started":"2023-04-27T07:05:17.957359Z","shell.execute_reply":"2023-04-27T07:05:17.964720Z"}}
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = in_dict[PAD_CHAR]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:18.149143Z","iopub.execute_input":"2023-04-27T07:05:18.149573Z","iopub.status.idle":"2023-04-27T07:05:18.158521Z","shell.execute_reply.started":"2023-04-27T07:05:18.149534Z","shell.execute_reply":"2023-04-27T07:05:18.157253Z"}}
def train(model, iterator, optimizer, criterion, clip):
	model.train()

	epoch_loss = 0

	#     for i, batch in enumerate(iterator):

	#         src = batch.src
	#         trg = batch.trg
	for i, (src, trg) in enumerate(iterator):
		optimizer.zero_grad()
		print("src shape", src.shape)
		print("target shape", trg.shape)
		output = model(src, trg)

		# trg = [trg len, batch size]
		# output = [trg len, batch size, output dim]

		output_dim = output.shape[-1]

		output = output[1:].view(-1, output_dim)
		trg = trg[1:].view(-1)

		# trg = [(trg len - 1) * batch size]
		# output = [(trg len - 1) * batch size, output dim]

		loss = criterion(output, trg)

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

		optimizer.step()

		epoch_loss += loss.item()

	return epoch_loss / len(iterator)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:18.402804Z","iopub.execute_input":"2023-04-27T07:05:18.403210Z","iopub.status.idle":"2023-04-27T07:05:18.945142Z","shell.execute_reply.started":"2023-04-27T07:05:18.403172Z","shell.execute_reply":"2023-04-27T07:05:18.943867Z"}}
N_EPOCHS = 10
CLIP = 1
train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T07:05:18.946197Z","iopub.status.idle":"2023-04-27T07:05:18.947019Z","shell.execute_reply.started":"2023-04-27T07:05:18.946725Z","shell.execute_reply":"2023-04-27T07:05:18.946756Z"}}
def evaluate(model, iterator, criterion):
	model.eval()

	epoch_loss = 0

	with torch.no_grad():
		for i, (src, trg) in enumerate(iterator):
			#             src = batch.src
			#             trg = batch.trg

			output = model(src, trg, 0)  # turn off teacher forcing

			# trg = [trg len, batch size]
			# output = [trg len, batch size, output dim]

			output_dim = output.shape[-1]

			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)

			# trg = [(trg len - 1) * batch size]
			# output = [(trg len - 1) * batch size, output dim]

			loss = criterion(output, trg)

			epoch_loss += loss.item()

	return epoch_loss / len(iterator)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.379067Z","iopub.status.idle":"2023-04-27T06:42:16.379818Z","shell.execute_reply.started":"2023-04-27T06:42:16.379543Z","shell.execute_reply":"2023-04-27T06:42:16.379574Z"}}
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.381744Z","iopub.status.idle":"2023-04-27T06:42:16.382752Z","shell.execute_reply.started":"2023-04-27T06:42:16.382448Z","shell.execute_reply":"2023-04-27T06:42:16.382481Z"}}
import time

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.384910Z","iopub.status.idle":"2023-04-27T06:42:16.385452Z","shell.execute_reply.started":"2023-04-27T06:42:16.385159Z","shell.execute_reply":"2023-04-27T06:42:16.385187Z"}}
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
	start_time = time.time()
	train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
	valid_loss = evaluate(model, valid_dataloader, criterion)
	end_time = time.time()
	epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), 'tut1-model.pt')

	print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.386758Z","iopub.status.idle":"2023-04-27T06:42:16.387295Z","shell.execute_reply.started":"2023-04-27T06:42:16.387002Z","shell.execute_reply":"2023-04-27T06:42:16.387029Z"}}
# in_batch, out_batch = generate_batch(train[0:32])
# # in_batch.reshape(in_batch[0].shape,-1)
# print(in_batch)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.388727Z","iopub.status.idle":"2023-04-27T06:42:16.389210Z","shell.execute_reply.started":"2023-04-27T06:42:16.388950Z","shell.execute_reply":"2023-04-27T06:42:16.388976Z"}}
# for input_batch, output_batch in  train_iter:
#     print(input_batch,output_batch)
# print(list(train_iter))
# print(df_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.390642Z","iopub.status.idle":"2023-04-27T06:42:16.391167Z","shell.execute_reply.started":"2023-04-27T06:42:16.390867Z","shell.execute_reply":"2023-04-27T06:42:16.390893Z"}}
# def generate_batch(data_batch):

# #     data_batch, target_batch = [], []
# #     max_len_input = max(len(x[0]) for x in data_batch)
# #     max_len_output = max(len(x[1]) for x in data_batch)
# #     padded_input = [torch.nn.utils.rnn.pad_sequence(torch.tensor(x[0]), (0, max_len_input - len(x[0]))).unsqueeze(0) for x in data_batch]
# #     padded_output = [torch.nn.utils.rnn.pad_sequence(torch.tensor(x[1]), (0, max_len_output - len(x[1]))).unsqueeze(0) for x in data_batch]
# #     print(padded_input)
#     print(padded_output)
#     return torch.cat(padded_input), torch.cat(padded_output)


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.393465Z","iopub.status.idle":"2023-04-27T06:42:16.393943Z","shell.execute_reply.started":"2023-04-27T06:42:16.393693Z","shell.execute_reply":"2023-04-27T06:42:16.393719Z"}}
char2idx = {chr(i): i for i in range(ord('a'), ord('z') + 1)}
print(char2idx)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.395334Z","iopub.status.idle":"2023-04-27T06:42:16.395846Z","shell.execute_reply.started":"2023-04-27T06:42:16.395582Z","shell.execute_reply":"2023-04-27T06:42:16.395610Z"}}
df_train.iloc[0]

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.397008Z","iopub.status.idle":"2023-04-27T06:42:16.397525Z","shell.execute_reply.started":"2023-04-27T06:42:16.397263Z","shell.execute_reply":"2023-04-27T06:42:16.397291Z"}}
PAD_IDX = out_dict[PAD_CHAR]
SOW_IDX = out_dict[SOW_CHAR]
EOW_IDX = out_dict[EOW_CHAR]


def generate_batch(data_batch):
	data_batch, target_batch = [], []
	max_len_input = max(len(x[0]) for x in data_batch)
	max_len_output = max(len(x[1]) for x in data_batch)
	padded_input = [torch.nn.utils.rnn.pad_sequence(torch.tensor(x[0]), (0, max_len_input - len(x[0]))).unsqueeze(0) for
					x in data_batch]
	padded_output = [torch.nn.utils.rnn.pad_sequence(torch.tensor(x[1]), (0, max_len_output - len(x[1]))).unsqueeze(0)
					 for x in data_batch]
	print(padded_input)
	print(padded_output)
	return torch.cat(padded_input), torch.cat(padded_output)


#     for (data_item, target_item) in data_batch:
#         data_batch.append(torch.cat([torch.tensor([SOW_IDX]), data_item, torch.tensor([EOW_IDX])], dim=0))
#         print(torch.cat([torch.tensor([SOW_IDX]), data_item, torch.tensor([EOW_IDX])], dim=0))
#         target_batch.append(torch.cat([torch.tensor([SOW_IDX]), target_item, torch.tensor([EOW_IDX])], dim=0))
#     data_batch = pad_sequence(data_batch, padding_value=PAD_IDX)
#     target_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
#     return data_batch, target_batch
#     for (data_item, target_item) in data_batch:
#         data_batch.append(torch.cat([torch.tensor([SOW_IDX]), data_item, torch.tensor([EOW_IDX])], dim=0))
#         print(torch.cat([torch.tensor([SOW_IDX]), data_item, torch.tensor([EOW_IDX])], dim=0))
#         target_batch.append(torch.cat([torch.tensor([SOW_IDX]), target_item, torch.tensor([EOW_IDX])], dim=0))
#     data_batch = pad_sequence(data_batch, padding_value=PAD_IDX)
#     target_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
#     return data_batch, target_batch

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.399270Z","iopub.status.idle":"2023-04-27T06:42:16.399802Z","shell.execute_reply.started":"2023-04-27T06:42:16.399490Z","shell.execute_reply":"2023-04-27T06:42:16.399517Z"}}
class MyDataset(Dataset):
	def __init__(self, input_data, output_data):
		self.input_data = input_data
		self.output_data = output_data

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.401472Z","iopub.status.idle":"2023-04-27T06:42:16.401963Z","shell.execute_reply.started":"2023-04-27T06:42:16.401711Z","shell.execute_reply":"2023-04-27T06:42:16.401739Z"}}
def collate_fn(batch):
	# Find maximum length of sequences in batch
	max_len_input = max(len(x[0]) for x in batch)
	max_len_output = max(len(x[1]) for x in batch)

	# Pad input and output sequences with zeros
	padded_input = [torch.nn.functional.pad(torch.tensor(x[0]), (0, max_len_input - len(x[0]))).unsqueeze(0) for x in
					batch]
	padded_output = [torch.nn.functional.pad(torch.tensor(x[1]), (0, max_len_output - len(x[1]))).unsqueeze(0) for x in
					 batch]

	# Stack padded sequences into tensors and return
	return torch.cat(padded_input), torch.cat(padded_output)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.403612Z","iopub.status.idle":"2023-04-27T06:42:16.404090Z","shell.execute_reply.started":"2023-04-27T06:42:16.403836Z","shell.execute_reply":"2023-04-27T06:42:16.403863Z"}}
input_data = [['apple', 'banana'], ['orange']]
output_data = [['red', 'yellow'], ['orange']]

my_dataset = MyDataset(input_data=input_data, output_data=output_data)
my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, collate_fn=collate_fn)

for batch_idx, (input_batch, output_batch) in enumerate(my_dataloader):
	print(f"Batch {batch_idx}:")
	print("Input data:", input_batch)
	print("Output data:", output_batch)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.405530Z","iopub.status.idle":"2023-04-27T06:42:16.406020Z","shell.execute_reply.started":"2023-04-27T06:42:16.405760Z","shell.execute_reply":"2023-04-27T06:42:16.405788Z"}}
train_iter = DataLoader(df_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.407254Z","iopub.status.idle":"2023-04-27T06:42:16.407727Z","shell.execute_reply.started":"2023-04-27T06:42:16.407479Z","shell.execute_reply":"2023-04-27T06:42:16.407505Z"}}
valid_iter = DataLoader(df_valid, batch_size=BATCH_SIZE,
						shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(df_test, batch_size=BATCH_SIZE,
					   shuffle=True, collate_fn=generate_batch)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.409182Z","iopub.status.idle":"2023-04-27T06:42:16.409694Z","shell.execute_reply.started":"2023-04-27T06:42:16.409439Z","shell.execute_reply":"2023-04-27T06:42:16.409467Z"}}
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
	def __init__(self, input_data, output_data):
		self.input_data = input_data
		self.output_data = output_data

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):
		return self.input_data[index], self.output_data[index]


import struct


def collate_fn(batch):
	# Find maximum length of sequences in batch
	max_len_input = max(len(x[0]) for x in batch)
	max_len_output = max(len(x[1]) for x in batch)

	# Pad input and output sequences with zeros
	inpu_batch = [
		torch.nn.functional.pad(torch.ByteTensor(bytearray(x[0])), (0, max_len_input - len(x[0]))).unsqueeze(0) for x in
		batch]
	padded_output = [
		torch.nn.functional.pad(torch.ByteTensor(bytearray(x[1])), (0, max_len_output - len(x[1]))).unsqueeze(0) for x
		in batch]
	padded_input = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True)
	print(padded_input)
	# Stack padded sequences into tensors and return
	return torch.cat(padded_input), torch.cat(padded_output)


# Example usage
input_data = [['apple', 'banana'], ['orange']]
output_data = [['red', 'yellow'], ['orange']]

my_dataset = MyDataset(input_data=input_data, output_data=output_data)
my_dataloader = DataLoader(dataset=my_dataset, batch_size=2, collate_fn=collate_fn)

for batch_idx, (input_batch, output_batch) in enumerate(my_dataloader):
	print(f"Batch {batch_idx}:")
	print("Input data:", input_batch)
	print("Output data:", output_batch)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.411140Z","iopub.status.idle":"2023-04-27T06:42:16.411656Z","shell.execute_reply.started":"2023-04-27T06:42:16.411385Z","shell.execute_reply":"2023-04-27T06:42:16.411410Z"}}
for data, targets in train_iter:
	print(data)

# %% [markdown]
# # Parameter Declaration

# %% [code] {"execution":{"iopub.status.busy":"2023-04-27T06:42:16.413046Z","iopub.status.idle":"2023-04-27T06:42:16.413555Z","shell.execute_reply.started":"2023-04-27T06:42:16.413304Z","shell.execute_reply":"2023-04-27T06:42:16.413331Z"}}
INPUT_DIM = len(in_dict)
OUTPUT_DIM = len(out_dict)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# %% [code]
