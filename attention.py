import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()        
    self.hidden_size = hidden_size
      
    
  def dot_score(self, hidden_state, encoder_states):
    return torch.sum(hidden_state * encoder_states, dim=2)
  
          
  def forward(self, hidden, encoder_outputs, mask):
      
    attn_scores = self.dot_score(hidden, encoder_outputs)
    
    # Transpose max_length and batch_size dimensions
    attn_scores = attn_scores.t()
    
    # Apply mask so network does not attend <pad> tokens        
    attn_scores = attn_scores.masked_fill(mask == 0, -1e5)
    
    # Return softmax over attention scores      
    return F.softmax(attn_scores, dim=1).unsqueeze(1)