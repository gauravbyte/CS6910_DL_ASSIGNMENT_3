from pre_processing import *

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='lstm'):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        else:
            raise ValueError("Invalid RNN type specified!")
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):  # for LSTM, return tuple of (hidden, cell)
            return hidden[0], hidden[1]
        else:  # for RNN and GRU, return hidden state only
            return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='lstm'):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        else:
            raise ValueError("Invalid RNN type specified!")
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden


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
        hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
RNN_TYPE = 'lstm'
pred_src = "$bindya|"
pred_trg = '$बिन्द्या|'
# %%
enc = Encoder(input_dim=INPUT_DIM, emb_dim=ENC_EMB_DIM, hid_dim=HID_DIM, n_layers=N_LAYERS, dropout=ENC_DROPOUT,rnn_type=RNN_TYPE)
dec = Decoder(output_dim=OUTPUT_DIM, emb_dim=DEC_EMB_DIM, hid_dim=HID_DIM, n_layers=N_LAYERS, dropout=DEC_DROPOUT,rnn_type=RNN_TYPE)

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
    epoch_acc = 0
    pbar = tqdm(iterator, desc="Training",position=0, leave=True)
    for i, (data, target) in enumerate(pbar):
        src = data
        trg = target
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]

        output_reshaped = output[1:].reshape(-1, output_dim)
        trg_reshaped = trg[1:].reshape(-1)

        loss = criterion(output_reshaped, trg_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
#         output = output[1:]
        preds = output.argmax(dim=2)
        preds=preds[1:]
        trg=trg[1:]
        if i%500==0:
            print("preds",preds)
            print("targets",trg)
        matches = torch.eq(preds, trg)
        columns_matches = torch.sum(matches, dim=0)
        num_matching_columns = torch.sum(columns_matches == trg.shape[0])
        acc = num_matching_columns / trg.shape[1]
#         if(i%100==0):
#             print("preds ",preds)
#             print("target",trg)
        epoch_acc += acc.item()
        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (i + 1), train_acc=epoch_acc / (i + 1))
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(iterator):
            src = data
            trg = target
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            preds = output.argmax(dim=1)
            acc = (preds == trg).float().mean()
            epoch_acc += acc.item()
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model,input_word,actual_output):
    data_pred = [[input_word,actual_output]]
    data_batch = DataLoader(data_pred, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
    iterator = data_batch
    with torch.no_grad():
        for i, (data, target) in enumerate(iterator):
            src = data
            trg = target
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output_reshaped = output[1:].reshape(-1, output_dim)
            trg_reshaped = trg[1:].reshape(-1)
            preds = output.argmax(dim=2)
            print("input word",input_word)
            print("actual word",actual_output)
            predicted_word = ""
            for i in preds:
                if i.item() in [0,1,2]:
                    continue
                predicted_word += predicted_word + output_indx_char[i.item()]
            print("predicted word",predicted_word)
    return preds


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
gbar = tqdm(range(1, N_EPOCHS + 1),position=1,leave=True, desc='Epochs', total=N_EPOCHS)
for epoch in gbar:
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, criterion)
    gbar.set_postfix(train_loss=train_loss, train_acc=train_accuracy, val_loss=valid_loss, val_acc=valid_accuracy)




predict(model,pred_src,pred_trg)
