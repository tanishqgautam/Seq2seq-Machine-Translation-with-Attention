import torch
import torch.nn as nn
from encdec import Encoder, Decoder

class seq2seq(nn.Module):
  def __init__(self, embedding_size, hidden_size, vocab_size, device, pad_idx, eos_idx, sos_idx):
    super(seq2seq, self).__init__()
    
    # Embedding layer shared by encoder and decoder
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    # Encoder network
    self.encoder = Encoder(hidden_size, 
                            embedding_size,
                            num_layers=2,
                            dropout=0.3)
    
    # Decoder network        
    self.decoder = Decoder(embedding_size,
                            hidden_size,
                            vocab_size,
                            n_layers=2,
                            dropout=0.3)
    
    
    # Indices of special tokens and hardware device 
    self.pad_idx = pad_idx
    self.eos_idx = eos_idx
    self.sos_idx = sos_idx
    self.device = device
      
  def create_mask(self, input_sequence):
    return (input_sequence != self.pad_idx).permute(1, 0)
      
      
  def forward(self, input_sequence, output_sequence):
    
    # Unpack input_sequence tuple
    input_tokens = input_sequence[0]
  
    # Unpack output_tokens, or create an empty tensor for text generation
    if output_sequence is None:
      inference = True
      output_tokens = torch.zeros((100, input_tokens.shape[1])).long().fill_(self.sos_idx).to(self.device)
    else:
      inference = False
      output_tokens = output_sequence[0]
    
    vocab_size = self.decoder.output_size
    batch_size = len(input_sequence[1])
    max_seq_len = len(output_tokens)
    
    # tensor to store decoder outputs
    outputs = torch.zeros(max_seq_len, batch_size, vocab_size).to(self.device)        
    
    # pass input sequence to the encoder
    encoder_outputs, hidden = self.encoder(input_tokens)
    
    # first input to the decoder is the <sos> tokens
    output = output_tokens[0,:]
    
    # create mask
    mask = self.create_mask(input_tokens)
    
    
    # Step through the length of the output sequence one token at a time
    for t in range(1, max_seq_len):
      output = output.unsqueeze(0)
      
      output, hidden = self.decoder(output, hidden, encoder_outputs, mask)
      outputs[t] = output
      
      if inference:
        output = output.max(1)[1]
      else:
        output = output_tokens[t]
      
      # If we're in inference mode, keep generating until we produce an
      # <eos> token
      if inference and output.item() == self.eos_idx:
        return outputs[:t]
        
    return outputs