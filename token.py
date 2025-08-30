import os, string, requests
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, token_to_idx, seq_len):
        self.seq_len = seq_len
        self.text = text
        self.encoded = [token_to_idx[c] for c in text]

    def __len__(self):
        return len(self.text) - self.seq_len
    def __getitem__(self, idx):
        x = self.encoded[idx:idx+self.seq_len]
        y = self.encoded[idx+1:idx+1+self.seq_len]
        return torch.tensor(x, dtype = torch.long), torch.tensor(y, dtype=torch.long)


def reading(seq_len):
    if not os.path.exists('sonnets.txt'):
        url ='https://raw.githubusercontent.com/girafe-ai/ml-course/22f_basic/homeworks/lab02_deep_learning/sonnets.txt'
        response = requests.get(url)
        with open('sonnets.txt', 'w') as file:
            file = response.text
    with open('sonnets.txt', 'r') as iofile:
        text = iofile.readlines()

    TEXT_START = 45
    TEXT_END = -368
    text = text[TEXT_START: TEXT_END]
    assert len(text) == 2616

    text = ''.join().lower()

    assert len(text) ==100225, 'Are you sure you have concatenated all the strings?'
    assert not any([x in set(text) for x in string.ascii_uppercase]), 'Uppercase letters are present'

    tokens = sorted(set(text))
    token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    idx_to_token = {i: tok for tok, i in token_to_idx.items()}



    train_data = CharDataset(text, token_to_idx, seq_len)
    val_data = CharDataset(text, token_to_idx, seq_len)

    return train_data, val_data




