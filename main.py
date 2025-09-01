#from my_token import CharDataset, reading
import my_token
import training
import models
import torch
#Hyperparameters
#_______________________________________________________________________
default_seq_line = 32
default_batch_size = 128
default_embedding_dim = 64
default_hidden_dim = 128
default_num_epochs = 10
default_lr = 0.01

default_list = [default_seq_line,
                         default_batch_size,
                         default_embedding_dim,
                         default_hidden_dim,
                         default_num_epochs,
                         default_lr]
default_text = [ 'seq_line',
                'batch_size',
                'embedding_dim',
                'hidden_dim',
                'num_epochs',
                'lr']


hyperparameters = []
print('Введи значения гиперпараметров: ')
for i in range(len(default_list)):
    try:
        user_input = input(f'{default_text[i]}: ')
        if user_input.strip() == "":
            user_input = default_list[i]
    except (KeyboardInterrupt, EOFError):
        user_input = default_list[i]
    hyperparameters.append(user_input)

seq_line, batch_size, embedding_dim, hidden_dim, num_epochs, lr = tuple(hyperparameters)
print(tuple(hyperparameters))
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