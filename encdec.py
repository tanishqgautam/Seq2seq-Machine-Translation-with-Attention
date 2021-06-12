import torch
import torch.nn as nn

from utils import SRC
from attention import Attention

class Encoder(nn.Module):
  
  def __init__(self, hidden_size, embedding_size, num_layers=2, dropout=0.3):
    
    super(Encoder, self).__init__()
    
    # Basic network params
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.num_layers = num_layers
    self.dropout = dropout
    
    # Embedding layer that will be shared with Decoder
    self.embedding = nn.Embedding(len(SRC.vocab), embedding_size)
    # GRU layer
    self.gru = nn.GRU(embedding_size, hidden_size,
                      num_layers=num_layers,
                      dropout=dropout)
      
  def forward(self, input_sequence):
      
    # Convert input_sequence to word embeddings
    embedded = self.embedding(input_sequence)
            
    outputs, hidden = self.gru(embedded)
    
    # The ouput of a GRU has shape -> (seq_len, batch, hidden_size)
    return outputs, hidden

class Decoder(nn.Module):
  def __init__(self, embedding_size, hidden_size, output_size, n_layers=2, dropout=0.3):
      
    super(Decoder, self).__init__()
    
    # Basic network params
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout
    self.embedding = nn.Embedding(output_size, embedding_size)
            
    self.gru = nn.GRU(embedding_size, hidden_size, n_layers, 
                      dropout=dropout)
    
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.attn = Attention(hidden_size)
      
  def forward(self, current_token, hidden_state, encoder_outputs, mask):
    
    # convert current_token to word_embedding
    embedded = self.embedding(current_token)
    
    # Pass through GRU
    gru_output, hidden_state = self.gru(embedded, hidden_state)
    
    # Calculate attention weights
    attention_weights = self.attn(gru_output, encoder_outputs, mask)
    
    # Calculate context vector (weigthed average)
    context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
    
    # Concatenate  context vector and GRU output
    gru_output = gru_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((gru_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    
    # Pass concat_output to final output layer
    output = self.out(concat_output)
    
    # Return output and final hidden state
    return output, hidden_state