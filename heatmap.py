from preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, cell_type, bidirectional=True,
                 batch_size=BATCH_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        # self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, hidden=None):
        self.batch_size = src.shape[1]
        embedded = self.dropout(self.embedding(src))
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)
        if self.bidirectional:
            # Split hidden and cell into two halves along the first dimension
            hidden_chunks = torch.chunk(hidden, 2, dim=0)
            hidden = 0.5 * (hidden_chunks[0] + hidden_chunks[1])

            if self.cell_type == "LSTM":
                cell_chunks = torch.chunk(cell, 2, dim=0)
                cell = 0.5 * (cell_chunks[0] + cell_chunks[1])

            # Compute the average of forward and backward outputs
            output = output.permute(2, 1, 0)
            output_chunks = torch.chunk(output, 2, dim=0)
            output = 0.5 * (output_chunks[0] + output_chunks[1])
            output = output.permute(2, 1, 0)

        if (self.cell_type == "LSTM"):
            return output, hidden, cell
        else:
            return output, hidden

    def initHidden(self):
        if self.bidirectional == True:
            return torch.zeros(2 * self.num_layers, self.batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)


# write decoder with attention


class Decoder_with_attention(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout, embedding_size, cell_type="LSTM",
                 batch_size=BATCH_SIZE):
        super(Decoder_with_attention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(self.hidden_size + self.embedding_size, MAX_LENGTH)
        self.attn_combine = nn.Linear(self.hidden_size + self.embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        if cell_type == 'RNN':
            self.rnn = nn.RNN(self.hidden_size, self.hidden_size, num_layers, dropout=dropout)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers, dropout=dropout)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers, dropout=dropout)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)
        self.batch_size = input.size(1)
        output = self.embedding(input).view(-1, self.batch_size, self.embedding_size)
        output = self.dropout(output)
        if self.cell_type == "LSTM":
            attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)), dim=1)
        else:
            attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))
        attn_applied = attn_applied.squeeze(1)
        # print("attention applied shape",attn_applied.shape)
        output = torch.cat((output[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        if self.cell_type == "LSTM":
            output, (hidden, cell) = self.rnn(output, (hidden[0], hidden[1]))
            return self.out(output[0]), hidden, cell, attn_weights
        else:
            output, hidden = self.rnn(output, hidden)
            return self.out(output[0]), hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5, heatmap=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.heatmap = heatmap
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg, heatmap=False):
        attention_weights = []
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_hidden = self.encoder.initHidden()
        # encoder_hidden = self.encoder.initHidden()
        if self.encoder.cell_type == 'LSTM':
            encoder_output, encoder_hidden, encoder_cell = self.encoder(src, encoder_hidden)
        else:
            encoder_output, encoder_hidden = self.encoder(src, encoder_hidden)
        input_decoder = trg[0, :]
        decoder_hidden = encoder_hidden

        if self.decoder.cell_type == 'LSTM':
            decoder_cell = encoder_cell

        for t in range(1, trg_len):
            if self.decoder.cell_type == 'LSTM':
                decoder_output, decoder_hidden, decoder_cell, attn_weights = self.decoder(input_decoder, (
                decoder_hidden, decoder_cell), encoder_output)

            else:
                decoder_output, decoder_hidden, attn_weights = self.decoder(input_decoder, decoder_hidden,
                                                                            encoder_output)
            # print("attention_weights",attn_weights)
            if (heatmap):
                attention_weights.append(attn_weights)
            outputs[t] = decoder_output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            input_decoder = trg[t] if teacher_force else top1

        return outputs, attention_weights


def get_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
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
    pbar = tqdm(iterator, desc="Training", position=0, leave=True)

    for i, (data, target) in enumerate(pbar):
        src = data.to(device)
        trg = target.to(device)
        optimizer.zero_grad()
        output, attn_weights = model(src, trg)
        # print("attention weights", attn_weights)
        # print("attention weights shape", attn_weights.shape)
        # attn_weightscpu = attn_weights.cpu().detach().numpy()
        # sns.heatmap(attn_weightscpu, cmap="YlGnBu", annot=False,fmt='.2f')
        # plt.show()
        output_dim = output.shape[-1]
        output_reshaped = output[1:].reshape(-1, output_dim)
        trg_reshaped = trg[1:].reshape(-1)
        loss = criterion(output_reshaped, trg_reshaped)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        acc = get_accuracy(output, trg)
        epoch_acc += acc
        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (i + 1), train_acc=epoch_acc / (i + 1))
        if True:
            break
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


from matplotlib.font_manager import FontProperties

font_prop = FontProperties(fname='Mangal 400.TTF')

# %%
def plot_attention_heatmap(model, iterator):
    for i, (data, target) in enumerate(iterator):
        src = data.to(device)
        trg = target.to(device)
        # get first 10 examples
        # src = data[:10].to(device)
        # trg = target[:10].to(device)
        output, attention_weights = model(src, trg, heatmap=True)
        # attention_weights = torch.stack(attention_weights).squeeze()  # Convert the list to a tensor
        # attention_weights = torch.cat(attention_weights, dim=0)
        # attention_weights = torch.stack(attention_weights).squeeze().cpu().detach().numpy()
        attention_weights = torch.stack(attention_weights, dim=0)  # Convert the list to a tensor

        attention_weights = attention_weights.squeeze().cpu().detach().numpy()  # Convert the tensor to a numpy array
        # attention_weights shape = (trg_len, batch_size, src_len)
        # attention_weights.permute(1,0,2)
        attention_weights = attention_weights.transpose(1, 0, 2)
        # attention_weights shape = (batch_size, trg_len, src_len)
        # print("attention weights shape",attention_weights.shape)
        # print("attention weights",attention_weights)
        # get first 10 examples
        attention_weights = attention_weights[:10]
        #convert attention weights to list
        attention_weights_list = attention_weights.tolist()
        # get first 10 examples
        # src shape = (src_len, batch_size)
        src = src.transpose(1, 0)
        trg = trg.transpose(1, 0)
        src = src[:10]
        trg = trg[:10]
        #convert source targe to list
        src_list = src.tolist()
        trg_list = trg.tolist()
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.tight_layout(pad=5.0)
        fig.subplots_adjust(top=0.90)
        axes = axes.ravel()

        for j in range(len(attention_weights)-1):

            output_vs_input_attention = attention_weights[j]
            max_src_len = src.shape[1]
            max_trg_len = trg.shape[1]

            for k in range(max_src_len):
                if src[j][k] == 0:
                    max_src_len = k + 1
                    break

            for k in range(max_trg_len):
                if trg[j][k] == 0:
                    max_trg_len = k + 1
                    break
            cur_src = src_list[j][:max_src_len]
            cur_trg = trg_list[j][:max_trg_len]
            output_vs_input_attention = output_vs_input_attention[:max_trg_len, :max_src_len]

            # Plot heatmap
            sns.heatmap(output_vs_input_attention, ax=axes[j], cmap="YlGnBu", cbar=False)

            # Set x-axis and y-axis labels
            axes[j].set_xlabel("Input")
            axes[j].set_ylabel("Output")

            # Set x-tick labels and rotate if needed
            input_characters = [INPUT_INDEX_CHAR[i] for i in cur_src]
            axes[j].set_xticks(range(len(cur_src)))
            axes[j].set_xticklabels(input_characters, rotation=90)

            # Set y-tick labels
            output_characters = [OUTPUT_INDEX_CHAR[i] for i in cur_trg]
            axes[j].set_yticks(range(len(cur_trg)))
            axes[j].set_yticklabels(output_characters, fontproperties=font_prop, fontdict={'fontsize': 9})

        fig.tight_layout()

        # Show the plot
        plt.show()

        if True:
            break


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    # with torch.no_grad():
    for i, (data, target) in enumerate(iterator):
        src = data.to(device)
        trg = target.to(device)
        output, _ = model(src, trg)
        output_dim = output.shape[-1]
        output_reshaped = output[1:].reshape(-1, output_dim)
        trg_reshaped = trg[1:].reshape(-1)
        loss = criterion(output_reshaped, trg_reshaped)
        acc = get_accuracy(output, trg)
        epoch_acc += acc
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, input_word, actual_output):
    data_pred = [[input_word, actual_output]]
    data_batch = DataLoader(data_pred, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
    iterator = data_batch

    src = data
    trg = target
    output = model(src, trg, 0)
    output_dim = output.shape[-1]
    # output_reshaped = output[1:].reshape(-1, output_dim)
    # trg_reshaped = trg[1:].reshape(-1)
    preds = output.argmax(dim=2)
    print("input word", input_word)
    print("actual word", actual_output)
    predicted_word = ""
    for i in preds:
        if i.item() in [0, 1, 2]:
            continue
        predicted_word += predicted_word + OUTPUT_INDEX_CHAR[i.item()]
    print("predicted word", predicted_word)
    return preds


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
BI_DIRECTION = True
CELL_TYPE = 'LSTM'
pred_src = "$bindya|"
pred_trg = '$बिन्द्या|'
BATCH_SIZE = 32
enc = Encoder(input_size=INPUT_DIM, embedding_size=ENC_EMB_DIM, hidden_size=HIDDEN_SIZE, num_layers=num_layers,
              dropout=ENC_DROPOUT, cell_type=CELL_TYPE, bidirectional=BI_DIRECTION)
dec = Decoder_with_attention(output_size=OUTPUT_DIM, embedding_size=DEC_EMB_DIM, hidden_size=HIDDEN_SIZE,
                             num_layers=num_layers,
                             dropout=DEC_DROPOUT, cell_type=CELL_TYPE, batch_size=BATCH_SIZE)

model = Seq2Seq(enc, dec, device, teacher_forcing_ratio=TEACHER_FORCING).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
best_valid_loss = float('inf')
gbar = tqdm(range(1, N_EPOCHS + 1), position=1, leave=True, desc='Epochs', total=N_EPOCHS)
for epoch in gbar:
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, criterion)
    gbar.set_postfix(train_loss=train_loss, train_acc=train_accuracy, val_loss=valid_loss, val_acc=valid_accuracy)

plot_attention_heatmap(model, valid_dataloader)

# predict(model, pred_src, pred_trg)
# predict(model,"$बिन्द्या|","$bindya|")
# %%
