from pre_processing import *


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, n_layers, dropout, rnn_type='lstm'):
        super(Encoder,self).__init__()
        self.hid_dim = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.iter = 0

    def forward(self, src):
        self.iter += 1
        print("Encoder forward",self.iter)
        embedded = self.dropout(self.embedding(src))
        encoder_states, (hidden, cell) = self.rnn(embedded)
        hidden = self.fc_hidden(torch.cat((hidden[0:1],hidden[1:2]),dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1],cell[1:2]),dim=2))
        return encoder_states, hidden, cell



class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim, n_layers)
        self.energy = nn.Linear(hid_dim * 3, 1)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, input, encoder_states, hidden, cell=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped,encoder_states),dim=2)))
        attention = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        rnn_input = torch.cat((context_vector, embedded), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc_out(output).squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of Encoder and Decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and Decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_states, hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input,encoder_states, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            #             input = trg[t]
            input = trg[t] if teacher_force else top1
        return outputs



INPUT_DIM = input_vocab_size
OUTPUT_DIM = output_vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
RNN_TYPE = 'rnn'
pred_src = "$bindya|"
pred_trg = '$बिन्द्या|'
# %%

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)

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
    epoch_acc = 0
    pbar = tqdm(iterator, desc="Training", position=0, leave=True)
    for i, (data, target) in enumerate(pbar):
        src = data
        trg = target
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        #         print(src.shape)
        #         print(trg.shape)

        output_reshaped = output.reshape(-1, output_dim)
        trg_reshaped = trg.reshape(-1)
        #         output_reshaped = output[1:].reshape(-1, output_dim)
        #         trg_reshaped = trg[1:].reshape(-1)
        preds = output.argmax(dim=2)
        #         trg_float = trg.float()
        loss = criterion(output_reshaped, trg_reshaped)
        #         loss = criterion(output_reshaped, trg_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        matches = torch.eq(preds, trg)
        columns_matches = torch.sum(matches, dim=0)
        num_matching_columns = torch.sum(columns_matches == trg.shape[0])
        acc = num_matching_columns / trg.shape[1]

        #         acc = (preds == trg).float().mean()
        #         acc=0
        if i % 1000 == 0:
            print("preds ", preds)
            print("trg   ", trg)
            print(acc)
        #         print(acc)
        #         print(preds==trg)
        #         if all(preds==trg) == True:
        #             acc = 1
        #
        epoch_acc += acc.item()
        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (i + 1), train_acc=epoch_acc / (i + 1))
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # pbar = tqdm(iterator,desc="VALIDATION", leave=False)
    with torch.no_grad():
        for i, (data, target) in enumerate(iterator):
            src = data
            trg = target
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output_reshaped = output.reshape(-1, output_dim)
            trg_reshaped = trg.reshape(-1)
            loss = criterion(output_reshaped, trg_reshaped)
            preds = output.argmax(dim=2)
            matches = torch.eq(preds, trg)
            columns_matches = torch.sum(matches, dim=0)
            num_matching_columns = torch.sum(columns_matches == trg.shape[0])
            acc = num_matching_columns / trg.shape[1]
            #             preds = output.argmax(dim=1)
            #             acc = (preds == trg).float().mean()
            epoch_acc += acc.item()
            epoch_loss += loss.item()
            # pbar.set_postfix(valiidation_loss=epoch_loss / (i + 1), validation_acc=epoch_acc / (i + 1))
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, input_word, actual_output):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    data_pred = [[input_word, actual_output]]
    data_batch = DataLoader(data_pred, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
    iterator = data_batch
    with torch.no_grad():
        for i, (data, target) in enumerate(iterator):
            src = data
            trg = target
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            #             output_reshaped = output[1:].reshape(-1, output_dim)
            #             trg_reshaped = trg[1:].reshape(-1)
            trg_float = trg.float()
            loss = criterion(output, trg)

            preds = output.argmax(dim=1)
            # print("preds",preds)
            # print("trg",trg)
            # T_acc = (preds == trg)
            # print("T_acc",T_acc)
            # if(T_acc):
            #     epoch_acc += 1
            epoch_loss += loss.item()
            # print("preds",preds.shape)
            # print(preds)

    return
    # return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 20
CLIP = 1
best_valid_loss = float('inf')
pbar = tqdm(range(1, N_EPOCHS + 1), position=3, leave=True, desc='Epochs', total=N_EPOCHS)
for epoch in pbar:
    # pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{N_EPOCHS}', unit=' batches')
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, criterion)
    # tqdm
    pbar.set_postfix(train_loss=train_loss, train_acc=train_accuracy, val_loss=valid_loss, val_acc=valid_accuracy)
    # tqdm.write(f'Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_accuracy*100:.2f}%')

# predict(model,pred_src,pred_trg)