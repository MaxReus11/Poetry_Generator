import torch.nn as nn
import torch


class Vanilla_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Vanilla_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.W_ih = nn.Linear(embedding_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, h_0=None):
        batch_size, seq_line = x.shape[:2]
        x_embedded = self.embedding(x)

        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        else:
            h_t = h_0

            outputs = []
            for t in range(seq_line):
                x_t = x_embedded[:,t,:]
                h_t = self.activation(self.W_ih(x_t)+self.W_hh(h_t))
                outputs.append(h_t.unsqueeze(1))
            outputs = torch.cat(outputs, dim=1)
            logits = self.output_layer(outputs)
            return logits, h_t

class Vanilla_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Vanilla_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.activation = nn.Tanh()
        self.sigma = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        self.W_f = nn.Linear(embedding_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)


        self.W_i = nn.Linear(embedding_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        self.U_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(embedding_dim, hidden_dim)

        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, x, h_0 = None, c_0 = None):
        batch_size, seq_len = x.shape[:2]
        x_embedded = self.embedding(x)

        if h_0 is None and c_0 is None: # Посмотреть логику, чтобы всегда был не None
            h_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        elif c_0 is None:
            c_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        elif h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t = h_0
            c_t = c_0

            outputs = []

            for t in range(seq_len):
                x_t = x_embedded[:, t, :]
                f_t = self.sigma(self.W_f(x_t)+self.U_f(h_t))
                i_t = self.sigma(self.W_i(x_t)+self.U_i(h_t))
                o_t = self.sigma(self.W_o(x_t) + self.U_o(h_t))

                c_c = self.activation(self.W_c(x_t)+self.U_c(h_t))
                c_t =  f_t*c_t + i_t*c_c

                h_t = o_t*self.activation(c_t)
                outputs.append(h_t.unsqueeze(1))

            outputs = torch.cat(outputs, dim = 1)
            logits = self.output_layer(outputs)
            return logits, h_t, c_t









