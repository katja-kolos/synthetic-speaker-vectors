# Noise-robust classifier for classifier guidance in diffusion models
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Auxiliary functions

# Function to embed timesteps
# copied from the timemlp.py 
# note: removed repeat
def timestep_embedding(timesteps, dim, max_period=10000, device='cuda:3'):
    # turn scalar timestep into a vector: smooth multi-scale encoding of time
    # timestep controls noise level
    # model behaves differently at early timesteps (almost noise) vs late timesteps (clear speaker vectors)
    # nearby timesteps should have similar representations
    # ...and (!) the model must be able to interpolate to unseen timesteps
    # solution: embeddings!

    # embedding idea: sinusoidal embeddings. 
    # Low-freq sinusoids capture coarse time info, high-freq sinusoids capture fine-grained differences
    # max_period controls the lowest frequency
    
    half = dim // 2 #split embedding into sin and cosine halfs
    # log-spaced frequencies:
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device)
    # timesteps[:, None]: shape [B, 1]
    # freqs[None]: shape [1, half]
    # args: [B, half]
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) #[B, dim]
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    # ensures smoothness across timesteps -- generalize across noise levels
    return embedding

# Function to initialize normalized noise
def _init_noise(batch_size, dim, seed=42):
    maxs, mins = 0.5, -0.5 # init random noise in the range of GST embeddings
    noise = torch.rand(batch_size, dim) * (maxs - mins) + mins
    # noise = torch.randn(batch_size, self.dim)
    return noise

# Classifier architecture (a bit more complicated than classifier of clean embeddings)
class NoiseRobustClassifier(nn.Module):
    def __init__(self, input_dim, time_dim=128, hidden_dim=512, dropout=0.1):
        # number of classes is hard-coded to 2 (genders: f / m)
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)  # 2 classes: male/female
        self.time_dim = time_dim #so that we later do timestep embedding in forward

    def forward(self, x_t, t):
        # we need to embed data (x_t) AND timestep (t)
        t_emb = timestep_embedding(t, self.time_dim)
        t_h = self.time_mlp(t_emb) # [B, hidden_dim]
        h = self.net(x_t) + t_h # fuse time; in the ResNet blocks, too, the timestep embedding shifts the hidden activations
        logits = self.head(h)
        return logits # these logits will be used at classifier guidance

# Training & Evaluation
def train_model(model, train_loader, val_loader, scheduler, epochs=20, lr=1e-3, device="cpu"):
    # note: compared to training in `gender_classifier.py`, we also have scheduler here (from DDPM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total = 0
        for X, y in train_loader:
            # X: shape [B, dim]
            X, y = X.to(device), y.to(device)

            # sample timesteps
            t = torch.randint(0, scheduler.config.num_train_timesteps, (X.size(0), ), device=device, dtype=torch.long)
            # add noise with the same scheduler as in DDPM
            # noise = torch.randn_like(X)
            noise = _init_noise(X.size(0), X.size(1)).to(device)
            x_t = scheduler.add_noise(X, noise, t).to(device)

            
            logits = model(x_t, t)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total += X.size(0)

        # Validation
        val_acc = evaluate(model, val_loader, scheduler, device=device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

def evaluate(model, loader, scheduler, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            # sample timesteps
            t = torch.randint(0, scheduler.config.num_train_timesteps, (X.size(0), ), device=device, dtype=torch.long)
            # add noise with the same scheduler as in DDPM
            # noise = torch.randn_like(X)
            noise = _init_noise(X.size(0), X.size(1)).to(device)
            x_t = scheduler.add_noise(X, noise, t).to(device)

            logits = model(x_t, t)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total
