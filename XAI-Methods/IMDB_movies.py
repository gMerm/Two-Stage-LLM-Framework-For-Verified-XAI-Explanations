import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from captum.attr import IntegratedGradients
import pandas as pd
import json

DATA_PATH = "../Use-Case-datasets/IMDB/"
OUTPUT_PATH = "../XAI-Methods-outputs/IMDB/"
EPOCHS = 270

# pick device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load dataset and preprocess
df = pd.read_csv(DATA_PATH + "IMDB_Dataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
train_texts = df['review'][:20000].tolist()                 # subset for faster training
train_labels = df['label'][:20000].tolist()
test_texts = df['review'][20000:22000].tolist()
test_labels = df['label'][20000:22000].tolist()

# tokenization and vocabulary
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])  # unk = unknown token for out-of-vocab words
vocab.set_default_index(vocab["<unk>"])

def encode(text):
    return [vocab[token] for token in tokenizer(text)]
max_len = 100

def pad_sequence(seq):
    return seq[:max_len] + [0] * (max_len - len(seq))

# prepare data tensors and move to device
X_train = torch.tensor([pad_sequence(encode(text)) for text in train_texts], dtype=torch.long).to(device)
y_train = torch.tensor(train_labels, dtype=torch.float).to(device)
X_test = torch.tensor([pad_sequence(encode(text)) for text in test_texts], dtype=torch.long).to(device)
y_test = torch.tensor(test_labels, dtype=torch.float).to(device)

# LSTM = Long Short-Term Memory
class InterpretableLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)     # more like a lookup table, integer to vector, so can be processed by LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)            # forget gate, input gate, output gate
        self.fc = nn.Linear(hidden_dim, 1)                                      # fully connected layer
        self.sigmoid = nn.Sigmoid()                                             # sigmoid because we do binary classification
        
    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        out = self.fc(hidden[-1])               # out = final memory state
        return self.sigmoid(out).squeeze()
    
    def forward_with_embeddings(self, embeddings):
        # Forward pass that takes embeddings directly instead of indices
        _, (hidden, _) = self.lstm(embeddings)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze()      # have 1 output


def interpret_with_embeddings(model, input_ids, vocab):
    
    model.train()                                                           # keep model in training mode

    with torch.no_grad():
        input_embeddings = model.embedding(input_ids)                       # get embeddings for input ids = sample
    
    input_embeddings = input_embeddings.detach().requires_grad_(True)       # model doesn't change the embeddings, allows IG to compute gradients
    
    baseline_embeddings = torch.zeros_like(input_embeddings)                # baseline = all-zero embedding
    
    # Create a wrapper function for IG that takes embeddings as input
    def model_wrapper(embeddings):
        # Ensure the model stays in training mode during forward pass
        model.train()
        return model.forward_with_embeddings(embeddings)
    
    # Apply Integrated Gradients on embeddings
    # https://captum.ai/docs/extension/integrated_gradients
    ig = IntegratedGradients(model_wrapper)
    attributions = ig.attribute(       # interpolate n_steps steps from baseline to input = token level prediction                                 
        input_embeddings, 
        baseline_embeddings,
        n_steps=180,                   # computes the integral approximation
        method='gausslegendre'
    )
    
    # Sum attributions across embedding dimensions to get token-level attributions
    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    return token_attributions

# Initialize model
model = InterpretableLSTMClassifier(len(vocab), 64, 64).to(device)
criterion = nn.BCELoss()    # binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward() 
    optimizer.step()
    
    # Calculate accuracy
    with torch.no_grad():
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_train).float().mean()
    
    print(f"Epoch {epoch}, Loss {loss.item():.4f}, Accuracy {accuracy.item():.4f}")

# Test model performance
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = (test_outputs > 0.5).float()
    test_accuracy = (test_predictions == y_test).float().mean()
    print(f"Test Accuracy: {test_accuracy.item():.4f}")

# Interpretation function using the embedding approach
def interpret(text):
    input_ids = torch.tensor([pad_sequence(encode(text))], dtype=torch.long).to(device)
    
    # Use the custom interpretation function
    attributions = interpret_with_embeddings(model, input_ids, vocab)
    
    tokens = [vocab.lookup_token(i) for i in input_ids[0].tolist()]
    return tokens, attributions

# Example interpretation
sample_id = 198
sample_text = test_texts[sample_id]
tokens, atts = interpret(sample_text)

with torch.no_grad():
    pred = model(torch.tensor([pad_sequence(encode(sample_text))], dtype=torch.long).to(device))

print("Full Review:", sample_text)
print("\nPrediction:", pred.item())
print("Predicted sentiment:", "Positive" if pred.item() > 0.5 else "Negative")

print("\nTop 20 Token Attributions:")
non_padding_indices = [i for i, token in enumerate(tokens) if token != '<unk>' and token != '0']
token_attr_pairs = [(tokens[i], atts[i]) for i in non_padding_indices[:20]]

for tok, score in token_attr_pairs:
    print(f"{tok:15s} -> {score:.6f}")
    
# save full review, prediction, predicted sentiment, top 20 token attributions to a json file
output_data = {
    "sample_number": sample_id,
    "full_review": sample_text,
    "prediction": float(pred.item()),
    "predicted_sentiment": "Positive" if pred.item() > 0.5 else "Negative",
    "top_20_token_attributions": [{"token": tok, "attribution": float(score)} for tok, score in token_attr_pairs]
}
with open(OUTPUT_PATH + f"int_gradients_sample{sample_id}_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

# Show most positive and negative attributions
# sorted_attrs = sorted(enumerate(atts), key=lambda x: x[1], reverse=True)
# print(f"\nMost Positive Attribution: {tokens[sorted_attrs[0][0]]:15s} -> {sorted_attrs[0][1]:.6f}")
# print(f"Most Negative Attribution: {tokens[sorted_attrs[-1][0]]:15s} -> {sorted_attrs[-1][1]:.6f}")


# -------------------------- If want to plot, uncomment below --------------------------
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from matplotlib.patches import Rectangle
# import numpy as np

# # Set font parameters
# rcParams["font.family"] = "Times New Roman"
# rcParams["font.size"] = 24

# # Load the integrated gradients output
# sample_id = 198

# with open(OUTPUT_PATH + f"int_gradients_sample{sample_id}_output.json", "r") as f:
#     data = json.load(f)

# # Extract all 20 token attributions
# token_attr_pairs = [(item['token'], item['attribution']) for item in data['top_20_token_attributions']]

# # Normalize attributions for color intensity
# tokens = [item[0] for item in token_attr_pairs]
# attributions = np.array([item[1] for item in token_attr_pairs])

# # Normalize to [-1, 1] range for color mapping
# max_abs = np.max(np.abs(attributions))
# normalized_attrs = attributions / max_abs if max_abs > 0 else attributions

# # Create figure
# fig, ax = plt.subplots(figsize=(12, 2.5))
# ax.axis('off')
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # Display tokens with background colors
# x_pos = 0.02
# y_pos = 0.85
# spacing = 0.01
# words_per_line = 7
# word_count = 0

# for token, norm_attr in zip(tokens, normalized_attrs):
#     # Force certain words to be negative with strong intensity
    
#     # Determine color based on attribution
#     if norm_attr > 0:
#         # Positive attribution - shades of blue
#         color = (1 - norm_attr * 0.7, 1 - norm_attr * 0.5, 1)
#     else:
#         # Negative attribution - shades of red
#         intensity = abs(norm_attr)
#         color = (1, 1 - intensity * 0.7, 1 - intensity * 0.7)
    
#     # Add text with background
#     text = ax.text(x_pos, y_pos, f' {token} ', 
#                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
#                             edgecolor='gray', linewidth=0.5),
#                    verticalalignment='center',
#                    fontsize=20)
    
#     # Update x position for next token
#     # Get text width in figure coordinates
#     fig.canvas.draw()
#     bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
#     bbox_fig = bbox.transformed(fig.transFigure.inverted())
#     x_pos += bbox_fig.width + spacing
    
#     word_count += 1
    
#     # Wrap to next line after 7 words
#     if word_count >= words_per_line:
#         x_pos = 0.02
#         y_pos -= 0.3
#         word_count = 0

# plt.tight_layout()
# plt.savefig(OUTPUT_PATH + f'IMDB_SaliencyMap.png', dpi=400, bbox_inches='tight', facecolor='white')
# plt.show()

# print(f"Prediction: {data['predicted_sentiment']} ({data['prediction']:.4f})")
# print(f"Saliency map saved to {OUTPUT_PATH}")