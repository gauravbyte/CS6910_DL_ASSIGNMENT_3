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
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


for data, target in train_dataloader:
    print(data.shape)
    print(target.shape)
    if True:
        break
