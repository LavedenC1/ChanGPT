import torch
import json
import re
from torch import nn

def load_model_and_vocab(model_path='language_model.pth', vocab_path='vocab.json'):
    print("Loading vocabulary and model...")
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    model = LSTMModel(len(vocab))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, vocab

def preprocess_input(text, vocab):
    tokenized = re.sub(r'[^\w\s]', '', text).lower().split()
    sequence = [vocab.get(word, vocab.get("<unk>")) for word in tokenized]
    return torch.tensor(sequence, dtype=torch.long).unsqueeze(0)

def generate_response(model, vocab, input_text):
    input_sequence = preprocess_input(input_text, vocab)
    with torch.no_grad():
        output = model(input_sequence)
    
    output = output.squeeze(0)
    output = output.argmax(dim=-1)

    predicted_tokens = [list(vocab.keys())[idx] for idx in output.tolist()]

    response = ' '.join(predicted_tokens)
    return response

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

if __name__ == '__main__':
    model_path = 'language_model.pth'
    vocab_path = 'vocab.json'
    
    model, vocab = load_model_and_vocab(model_path, vocab_path)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        response = generate_response(model, vocab, user_input)
        print("AI:", response)
