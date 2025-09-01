#from my_token import CharDataset, reading
import my_token
import training
import models
import torch
#Hyperparameters
#_______________________________________________________________________
print('Введи значения гиперпараметров: ')
print('seq_line - ')
seq_line = int(input())
print('batch_size - ')
batch_size = int(input())
print('embedding_dim - ')
embedding_dim = int(input())
print('hidden_dim - ')
hidden_dim = int(input())
print('num_epochs - ')
num_epochs = int(input())
print('lr - ')
lr = float(input())

#______________________________________________________________________

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Используется GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():  # для Mac M1/M2, не в Colab
    device = torch.device("mps")
    print("Используется Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("Используется CPU")

print("Текущий device:", device)


train_data, test_data, vocab_size = my_token.reading(seq_line)

model = models.Vanilla_RNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

training.training(model, train_data, test_data, batch_size, num_epochs, device, criterion, optimizer, vocab_size)