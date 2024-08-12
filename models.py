import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,
                          hidden_size, 
                          num_layers, 
                          batch_first = True, 
                          bidirectional = False, 
                          dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):    
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, hn = self.rnn(x, h0)
        # out_size = [batch_size, sequence_length, features]
        # hn_size = [num_layers, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * features]
        out = self.fc1(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        # out_size = [batch_size, 1]
        return out    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, dropout, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, num_layers, 
                            batch_first = True, 
                            bidirectional = False,
                            dropout = dropout)
        self.fc1 = nn.Linear(hidden_size * sequence_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x_size = [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out_size = [batch_size, sequence_length, hidden_size]
        # hn_size = [num_layers, batch_size, hidden_size]
        # cn_size = [num_layers, batch_size, hidden_size]
        out = out.reshape(out.shape[0], -1)
        # out_size = [batch_size, sequence_length * hidden_size]
        out = self.fc1(out)
        # out_size = [batch_size, 128]
        out = self.relu(out)
        out = self.fc2(out)
        # out_size = [batch_size, 64]
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        # out_size = [batch_size, 1]
        return out
  
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout, device):
        super(CNN_LSTM, self).__init__()
        self.device = device
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_size, 
                      out_channels = 64, 
                      kernel_size = kernel_size, 
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            nn.Conv1d(in_channels = 64,
                      out_channels = 128,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
        )
        self.lstm = nn.LSTM(input_size = 128,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # cnn input size = [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        # lstm input size = [batch_size, seq_len, input_size]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out