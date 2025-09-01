from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from torch import no_grad
def training(model, train_data, test_data, batch_size, num_epochs, device, criterion, optimizer, vocab_size):

    train_batches = DataLoader(train_data,  batch_size = batch_size, shuffle = True, drop_last=True)
    test_batches = DataLoader(test_data,  batch_size = batch_size, shuffle = True, drop_last=True)

    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for X_batch, Y_batch in train_batches:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits, _ = model(X_batch)

            loss = criterion(logits.view(-1, vocab_size), Y_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_epoch_train_loss = epoch_train_loss/len(train_batches)
        train_loss.append(avg_epoch_train_loss)
        model.eval()
        epoch_val_loss = 0
        with no_grad():
            for X_batch, Y_batch in test_batches:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                logits, _ = model(X_batch)

                val_loss = criterion(logits.view(-1, vocab_size), Y_batch.view(-1))

                epoch_val_loss += val_loss.item()

        avg_epoch_test_loss = epoch_val_loss / len(test_batches)
        test_loss.append(avg_epoch_test_loss)

        print(f"Epoch: {epoch+1:2d} | Training loss: {avg_epoch_train_loss:.4f} | Validation lost: {avg_epoch_test_loss:.4f}")
        clear_output(wait=True)

        plt.plot(range(1, epoch+2), train_loss, c='blue', label = 'Train')
        plt.plot(range(1, epoch+2), test_loss, c='orange', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)
        plt.show()

