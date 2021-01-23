import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data as d

import warnings
warnings.filterwarnings("ignore")

from utils import SRC, TRG, nlp_ru,fields
from seq2seq import seq2seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = d.TabularDataset(path="dataset/train.csv", format='csv', fields=fields)

SRC.build_vocab(data, max_size=4000)
TRG.build_vocab(data, max_size=4000)

train_data, val_data = data.split(split_ratio=0.8)
train_iterator, valid_iterator = d.BucketIterator.splits(
    (train_data, val_data), 
    batch_size = 64, 
    sort_within_batch = True, 
    sort_key = lambda x:len(x.rus),
    device = device)

# extract special tokens
pad_idx = TRG.vocab.stoi['<pad>']
eos_idx = TRG.vocab.stoi['<eos>']
sos_idx = TRG.vocab.stoi['<sos>']

# Size of embedding_dim should match the dim of pre-trained word embeddings!
embedding_dim = 100
hidden_dim = 256
vocab_size = len(TRG.vocab)

model = seq2seq(embedding_dim,
                hidden_dim, 
                vocab_size, 
                device, pad_idx, eos_idx, sos_idx).to(device)

optimizer = optim.Adam(model.parameters())

# cross entropy loss with softmax
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

def train(model, iterator, criterion, optimizer):
  # Put the model in training mode!
  model.train()
  
  epoch_loss = 0
  i = 0
  for (idx, batch) in enumerate(iterator):
      
    print("start", i)
    input_sequence = batch.rus
    output_sequence = batch.eng

    target_tokens = output_sequence[0]

    # zero out the gradient for the current batch
    optimizer.zero_grad()

    # Run the batch through our model
    output = model(input_sequence, output_sequence)

    # Throw it through our loss function
    output = output[1:].view(-1, output.shape[-1])
    target_tokens = target_tokens[1:].view(-1)

    loss = criterion(output, target_tokens)

    # Perform back-prop and calculate the gradient of our loss function
    loss.backward()

    # Update model parameters
    optimizer.step()

    epoch_loss += loss.item()
    i+= 1
      
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  # Put the model in training mode!
  model.eval()
  
  epoch_loss = 0
  
  for (idx, batch) in enumerate(iterator):
    input_sequence = batch.rus
    output_sequence = batch.eng

    target_tokens = output_sequence[0]

    # Run the batch through our model
    output = model(input_sequence, output_sequence)

    # Throw it through our loss function
    output = output[1:].view(-1, output.shape[-1])
    target_tokens = target_tokens[1:].view(-1)

    loss = criterion(output, target_tokens)

    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 1

best_valid_loss = float('inf')

# start model training
for epoch in range(N_EPOCHS):
    
  start_time = time.time()
  
  train_loss = train(model, train_iterator, criterion, optimizer)
  valid_loss = evaluate(model, valid_iterator, criterion)
  
  end_time = time.time()
  
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
  # compare validation loss
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'models/best_model.pt')
  
  print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def translate_sentence(model, sentence):
    model.eval()
    
    # tokenization
    tokenized = nlp_ru(sentence) 
    # convert tokens to lowercase
    tokenized = [t.lower_ for t in tokenized]
    # convert tokens to integers
    int_tokenized = [SRC.vocab.stoi[t] for t in tokenized] 
    
    # convert list to tensor
    sentence_length = torch.LongTensor([len(int_tokenized)]).to(model.device) 
    tensor = torch.LongTensor(int_tokenized).unsqueeze(1).to(model.device) 
    
    # get predictions
    translation_tensor_logits = model((tensor, sentence_length), None) 
    
    # get token index with highest score
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    # convert indices (integers) to tokens
    translation = [TRG.vocab.itos[t] for t in translation_tensor]
 
    # Start at the first index.  We don't need to return the <sos> token...
    translation = translation[1:]
    return " ".join(translation)


sentence = "Оно работает"
response = translate_sentence(model, sentence)
print(response)