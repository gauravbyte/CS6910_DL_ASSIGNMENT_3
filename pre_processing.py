import os
from tqdm.auto import tqdm
from datetime import time
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
from torch.utils.data import Dataset


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
PAD_CHAR = '_'
EOW_CHAR = '|'
SOW_CHAR = '$'
BATCH_SIZE = 32
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
output_char_indx = {PAD_CHAR: 0, EOW_CHAR: 1, SOW_CHAR: 2}
for index, alpha in enumerate(hindi_alphabets):
    output_char_indx[alpha] = index + 3
# %%
output_indx_char = {v: k for k, v in output_char_indx.items()}
print(output_char_indx)
print(len(output_char_indx))
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
output_vocab_size = len(output_char_indx)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()

# %%
input_vocab_size = len(in_dict)
output_vocab_size = len(output_char_indx)
print("Input Character max", input_vocab_size)
print("output Character size", output_vocab_size)

train_list = df_train.values.tolist()
valid_list = df_valid.values.tolist()
test_list = df_test.values.tolist()


# %% md
class Transliterate(Dataset):
    def __init__(self, df_data, in_dict, out_dict):
        super().__init__()
        self.df_data = df_data
        self.in_dict = in_dict
        self.out_dict = out_dict
        self.df_data = df_data.iloc[:, ].apply(lambda x: SOW_CHAR + x + EOW_CHAR)


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


train_data = Transliterate(df_train, in_dict, output_char_indx)
valid_data = Transliterate(df_valid, in_dict, output_char_indx)
test_data = Transliterate(df_test, in_dict, output_char_indx)
# %%
def inputToTensor(line):
    tensor = torch.tensor(data=([in_dict[x] for x in line]), dtype=torch.long)
    return tensor


def charToTensor(char, dic=in_dict):
    tensor = torch.zeros(len(dic))
    tensor[dic[char]] = 1
    return tensor


def outToTensor(word):
    tensor = torch.tensor([output_char_indx[x] for x in word])
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


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


for data, target in train_dataloader:
    print(data.shape)
    print(target.shape)
    if True:
        break
