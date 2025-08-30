import token
import training
import models
import torch
#Hyperparameters
#_______________________________________________________________________
vocab_size = 50
seq_line = 50
batch_size = 256
embedding_dim = 32
hidden_dim = 32
num_epochs = 10
device = torch.device('cuda')
lr = 0.1

#______________________________________________________________________

train_data, test_data = token.reading(seq_line)

model = models.Vanilla_RNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

training.training(model, train_data, test_data, batch_size, num_epochs, device, criterion, optimizer, vocab_size)