import torch
import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        epoch_acc += acc
        epoch_loss += loss.item()
        pbar.set_postfix(train_loss=epoch_loss / (i + 1), train_acc=epoch_acc / (i + 1))
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


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
 
# def predict(model,input_word,actual_output):
#     data_pred = [[input_word,actual_output]]
#     data_batch = DataLoader(data_pred, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
#     iterator = data_batch

#     src = data
#     trg = target
#     output = model(src, trg, 0)
#     output_dim = output.shape[-1]
#     # output_reshaped = output[1:].reshape(-1, output_dim)
#     # trg_reshaped = trg[1:].reshape(-1)
#     preds = output.argmax(dim=2)
#     print("input word",input_word)
#     print("actual word",actual_output)
#     predicted_word = ""
#     for i in preds:
#         if i.item() in [0,1,2]:
#             continue
#         predicted_word += predicted_word + OUTPUT_INDEX_CHAR[i.item()]
#     print("predicted word",predicted_word)
#     return preds