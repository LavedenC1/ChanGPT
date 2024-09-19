import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from collections import Counter
import re
from torch.nn.utils.rnn import pad_sequence
def load_data(json_file):
    print("Loading data from", json_file)
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    sentences = []
    
    if isinstance(data, list):
        for thread in data:
            replies = thread.get("replies", [])
            cleaned_replies = clean_replies(replies)
            sentences.extend(cleaned_replies)
    elif isinstance(data, dict):
        replies = data.get("replies", [])
        cleaned_replies = clean_replies(replies)
        sentences.extend(cleaned_replies)

    print(f"Loaded {len(sentences)} sentences.")
    return sentences

def clean_replies(replies):
    cleaned = []
    for reply in replies:
        reply = reply.strip()
        if reply and not reply.startswith(">>"):
            cleaned.append(reply)
    return cleaned

def tokenize(sentences):
    return [re.sub(r'[^\w\s]', '', sentence).lower().split() for sentence in sentences]  # Remove punctuation, split words

def build_vocab(tokenized_sentences):
    print("Building vocabulary...")
    word_counts = Counter([word for sentence in tokenized_sentences for word in sentence])
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=2)}  # Start from 2 to leave room for <pad> and <unk>
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def sentences_to_sequences(tokenized_sentences, vocab):
    print("Converting sentences to sequences...")
    sequences = [[vocab.get(word, vocab["<unk>"]) for word in sentence] for sentence in tokenized_sentences]
    print(f"Converted {len(sequences)} sequences.")
    return sequences

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

def collate_fn(batch):
    sequences = pad_sequence(batch, batch_first=True, padding_value=0)
    targets = sequences[:, 1:]
    inputs = sequences[:, :-1]
    return inputs, targets

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
def train_model(json_file, epochs=10, batch_size=32, embedding_dim=128, hidden_dim=128):
    print("Starting training...")
    sentences = load_data(json_file)
    tokenized_sentences = tokenize(sentences)
    vocab = build_vocab(tokenized_sentences)
    sequences = sentences_to_sequences(tokenized_sentences, vocab)
    dataset = TextDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = LSTMModel(len(vocab), embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}')
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader)}')
    torch.save(model.state_dict(), 'language_model.pth')
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)

    print('Training completed. Model and vocab saved.')

if __name__ == '__main__':
    json_file = 'data.json'
    train_model(json_file)
