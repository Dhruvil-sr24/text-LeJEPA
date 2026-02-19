import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from main import TextLeJEPA
# ==========================================
# 1. ARCHITECTURE WRAPPER (For Layer Extraction)
# ==========================================
class TextLeJEPAProber:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    # @torch.no_grad()
    # def extract_all_layers(self, dataloader):
        # """Extracts pooled embeddings for every transformer layer."""
        # # Initialize storage: {Layer_Index: [List of Tensors]}
        # # Layer 0 is the embedding layer; 1 to N are transformer layers
        # num_layers = len(self.model.encoder.layers)
        # layer_data = {i: [] for i in range(num_layers + 1)}
        # all_labels = []

        # print(f"Starting feature extraction from {num_layers} layers...")
        
        # for batch in dataloader:
        #     input_ids = batch["input_ids"].to(self.device)
        #     attention_mask = batch["attention_mask"].to(self.device)
        #     labels = batch["labels"]
            
        #     # 1. Get Initial Embeddings (Layer 0)
        #     x = self.model.token_embedding(input_ids)
        #     x = x + self.model.pos_embedding[:, :input_ids.shape[1], :]
            
        #     # Masked Mean Pooling helper
        #     mask = attention_mask.unsqueeze(-1)
        #     def pool(h): return (h * mask).sum(dim=1) / mask.sum(dim=1)

        #     layer_data[0].append(pool(x).cpu())

        #     # 2. Pass through Transformer Layers
        #     current_h = x
        #     # We iterate manually through the layers stored in the encoder
        #     for i, layer in enumerate(self.model.encoder.layers):
        #         current_h = layer(current_h)
        #         layer_data[i+1].append(pool(current_h).cpu())
            
        #     all_labels.append(labels.cpu())

        # # Concatenate lists into single tensors
        # final_features = {l: torch.cat(tensors) for l, tensors in layer_data.items()}
        # final_labels = torch.cat(all_labels)
        
        # return final_features, final_labels
    @torch.no_grad()
    def extract_all_layers(self, dataloader):
        """Extracts pooled embeddings for every transformer layer."""
        num_layers = len(self.model.encoder.layers)
        layer_data = {i: [] for i in range(num_layers + 1)}
        all_labels = []

        print(f"Starting feature extraction from {num_layers} layers...")
        
        for batch in dataloader:
            # Handle both Dictionary (HuggingFace style) and List (TensorDataset style)
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]
            else:
                # Based on: TensorDataset(dummy_input, dummy_mask, dummy_labels)
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            
            # 1. Get Initial Embeddings (Layer 0)
            x = self.model.token_embedding(input_ids)
            x = x + self.model.pos_embedding[:, :input_ids.shape[1], :]
            
            # Masked Mean Pooling helper
            mask = attention_mask.unsqueeze(-1)
            def pool(h): return (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

            layer_data[0].append(pool(x).cpu())

            # 2. Pass through Transformer Layers
            current_h = x
            for i, layer in enumerate(self.model.encoder.layers):
                current_h = layer(current_h)
                layer_data[i+1].append(pool(current_h).cpu())
            
            all_labels.append(labels.cpu())

        # Concatenate lists into single tensors
        final_features = {l: torch.cat(tensors) for l, tensors in layer_data.items()}
        final_labels = torch.cat(all_labels)
        
        return final_features, final_labels
# ==========================================
# 2. PROBING ENGINE
# ==========================================
def run_comprehensive_probe(layer_features, labels, num_classes, batch_size=256, epochs=30, device="cuda"):
    results = {}
    
    for layer_idx, X in layer_features.items():
        print(f"\n--- Training Probe on Layer {layer_idx} ---")
        
        # Split Data
        n_train = int(0.8 * len(X))
        train_loader = DataLoader(TensorDataset(X[:n_train], labels[:n_train]), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X[n_train:], labels[n_train:]), batch_size=batch_size)

        # Define Linear Probe
        probe = nn.Linear(X.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        probe.train()
        for epoch in range(epochs):
            for b_X, b_y in train_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                logits = probe(b_X)
                loss = criterion(logits, b_y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Evaluation
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b_X, b_y in test_loader:
                b_X, b_y = b_X.to(device), b_y.to(device)
                preds = probe(b_X).argmax(1)
                correct += (preds == b_y).sum().item()
                total += b_y.size(0)
        
        acc = correct / total
        results[layer_idx] = acc
        print(f"Layer {layer_idx} Accuracy: {acc:.4f}")

    return results

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION (Match your training config) ---
    CONFIG = {
        'vocab_size': 30522,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'max_seq_len': 128,
        'predictor_depth': 2,
        'mask_ratio': 0.5,
        'checkpoint_path': '/teamspace/studios/this_studio/checkpoints/text_lejepa_best.pt'
    }
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model
    model = TextLeJEPA(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        max_seq_len=CONFIG['max_seq_len'],
        predictor_depth=CONFIG['predictor_depth'],
        mask_ratio=CONFIG['mask_ratio']
    )

    # 2. Load Checkpoint (Handling the dictionary structure)
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # 3. Prepare Data (Dummy data example - replace with your real DataLoader)
    # Using small dummy set for script completeness
    dummy_input = torch.randint(0, CONFIG['vocab_size'], (1000, CONFIG['max_seq_len']))
    dummy_mask = torch.ones((1000, CONFIG['max_seq_len']))
    dummy_labels = torch.randint(0, 2, (1000,)) # Binary classification
    
    probing_loader = DataLoader(
        TensorDataset(dummy_input, dummy_mask, dummy_labels), 
        batch_size=32
    )
    # Note: Wrap your actual DataLoader to return dict keys "input_ids", "attention_mask", "labels"
    # Or modify the loop in extract_all_layers to match your dataloader output.

    # 4. Extract & Probe
    prober = TextLeJEPAProber(model, device=DEVICE)
    
    # We redefine the loop slightly here if your loader isn't using dicts
    # In a real scenario, use your ProbingDataset class.
    
    print("Extracting features...")
    features, labels = prober.extract_all_layers(probing_loader)

    print("Running Probes...")
    results = run_comprehensive_probe(features, labels, num_classes=2, device=DEVICE)

    # 5. Visualize
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), marker='o', color='royalblue')
    plt.xlabel("Layer Index (0=Embeddings)")
    plt.ylabel("Linear Probe Accuracy")
    plt.title("JEPA Layer-wise Feature Quality")
    plt.grid(True, alpha=0.3)
    plt.show()