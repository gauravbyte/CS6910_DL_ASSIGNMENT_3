import torch
from torch import nn

# if 'PYTHONPATH' not in os.environ:
#     if 'kaggle' not in os.environ['PYTHONPATH']:
#         from pre_processing import *


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
