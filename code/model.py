import torch


class LSTM(torch.nn.Module):
  def __init__(self,input_size,hidden_size,layers):
    super().__init__()
    self.layers = layers
    self.hidden_size = hidden_size
    self.lstm = torch.nn.LSTM(input_size,hidden_size,layers,batch_first=True)
    self.fc = torch.nn.Linear(hidden_size,1)

  def forward(self,X):
    batch_size = X.size(0)
    h0 = torch.zeros(self.layers,batch_size,self.hidden_size)
    c0 = torch.zeros(self.layers,batch_size,self.hidden_size)
    out, _ = self.lstm(X, (h0,c0))
    return self.fc(out[:,-1,:])


