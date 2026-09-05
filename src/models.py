from typing import Dict, List
import glob
import os
import pickle
import re
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from astra import *

MODELS_DIR = "models"

class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMTickClassifier:
    def __init__(self, seq_len=10, input_dim=4, hidden_dim=32, lr=0.01, epochs=20, batch_size=32):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.classes_ = np.array([0, 1])
        self.model = _LSTMNet(input_dim, hidden_dim)
        self.lr = lr

    def fit(self, X, y):
        # Convert 2D (samples, features) to 3D (samples, seq_len, features)
        num_samples = len(X) - self.seq_len + 1
        if num_samples <= 0:
            raise ValueError(f"Need at least {self.seq_len} samples to train LSTM.")
            
        seqs = np.array([X[i : i + self.seq_len] for i in range(num_samples)])
        targets = np.array(y[self.seq_len - 1 :])
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(seqs, dtype=torch.float32), 
            torch.tensor(targets, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = Astra(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                def closure():
                    optimizer.zero_grad()
                    loss = criterion(self.model(batch_x), batch_y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_arr = np.asarray(X)
        
        # Handle real-time single prediction from the dashboard/pipeline
        if X_arr.ndim == 2:
            if len(X_arr) < self.seq_len:
                padding = np.repeat(X_arr[:1], self.seq_len - len(X_arr), axis=0)
                X_arr = np.vstack([padding, X_arr])
            X_tensor = torch.tensor(X_arr[-self.seq_len:], dtype=torch.float32).unsqueeze(0)
        else:
            X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(self.model(X_tensor), dim=-1).cpu().numpy()
        return probs if probs.ndim == 2 else probs[np.newaxis, :]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)


def train_models(X: List[List[float]], y: List[int]) -> Dict[str, object]:
    """
    Train Logistic Regression, Random Forest, and LSTM on the bootstrap dataset.
    """
    if not X or not y:
        raise ValueError("train_models requires a non-empty X and y")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if len(set(y)) < 2:
        raise ValueError(
            "Need at least two label classes (0 and 1) to train a classifier; "
            f"got only {set(y)}. Collect more bootstrap data."
        )

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X, y)

    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
    random_forest.fit(X, y)
    
    lstm_model = LSTMTickClassifier(seq_len=10, input_dim=len(X[0]))
    lstm_model.fit(X, y)

    return {"logreg": logreg, "random_forest": random_forest, "lstm": lstm_model}
def predict(models: Dict[str, object], feature_vector: List[float]) -> Dict[str, int]:
    """
    Get a prediction (0 or 1) from each trained model for one feature vector.
    """
    if not models:
        raise ValueError("predict requires a non-empty models dict")

    vector_2d = [list(feature_vector)]
    predictions: Dict[str, int] = {}
    for name, model in models.items():
        predictions[name] = int(model.predict(vector_2d)[0])
    return predictions


def _next_version(name: str) -> int:
    """
    Scans MODELS_DIR for files like '{name}_v<N>_<timestamp>.pkl' and
    returns the next version number to use (1 if none exist yet).
    """
    pattern = os.path.join(MODELS_DIR, f"{name}_v*_*.pkl")
    existing = glob.glob(pattern)
    version_re = re.compile(rf"^{re.escape(name)}_v(\d+)_")

    versions = []
    for path in existing:
        match = version_re.match(os.path.basename(path))
        if match:
            versions.append(int(match.group(1)))

    return max(versions, default=0) + 1


def save_models(models: Dict[str, object]) -> Dict[str, str]:
    """
    Save each trained model to disk with a versioned filename.
    """
    if not models:
        raise ValueError("save_models requires a non-empty models dict")

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    saved_paths: Dict[str, str] = {}
    for name, model in models.items():
        version = _next_version(name)
        filename = f"{name}_v{version}_{timestamp}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        saved_paths[name] = filepath

    return saved_paths