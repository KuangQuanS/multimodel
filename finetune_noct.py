import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR

class ModalityEncoder(nn.Module):
    """Simplified encoder for single modality"""
    def __init__(self, dim_in, dim_latent=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_latent),
            nn.BatchNorm1d(dim_latent)
        )

    def forward(self, x):
        return self.encoder(x)

def load_encoder(mod, checkpoint_dir, latent_dim):
    """Load pretrained encoder"""
    ckpt = torch.load(os.path.join(checkpoint_dir, f"{mod}_best.pth"),
                     map_location="cpu", weights_only=True)
    dim_in = ckpt['0.weight'].shape[1]  # Get input dim from first layer weights
    encoder = ModalityEncoder(dim_in=dim_in, dim_latent=latent_dim)
    encoder.encoder.load_state_dict(ckpt)
    return encoder

class MLPFusion(nn.Module):
    """MLP-based fusion module"""
    def __init__(self, dim_latent, n_modalities, num_classes=2, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim_latent * n_modalities, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, zs):
        # zs: [B, M, D]
        B, M, D = zs.shape
        zs_flat = zs.view(B, -1)  # Flatten all modalities [B, M*D]
        return self.fusion_mlp(zs_flat)

class MultiModalDataset(Dataset):
    def __init__(self, npz_path, modalities, indices=None):
        data = np.load(npz_path)
        self.Xs = [data[mod] for mod in modalities]
        self.y = data['y']
        if indices is not None:
            self.Xs = [x[indices] for x in self.Xs]
            self.y = self.y[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [torch.FloatTensor(x[idx]) for x in self.Xs], torch.tensor(self.y[idx], dtype=torch.long)

def train_epoch(model, encoders, loader, optimizer, criterion, device, l1_lambda=0.0):
    model.train()
    total = correct = 0
    for Xlist, yb in loader:
        yb = yb.to(device)
        
        # Process each modality through its encoder
        zs = torch.stack([
            encoders[mod](x.to(device)) 
            for mod, x in zip(encoders.keys(), Xlist)
        ], dim=1)  # [B, M, D]
        
        logits = model(zs)
        loss = criterion(logits, yb)
        
        # L1 regularization
        if l1_lambda > 0:
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()
    
    return correct / total * 100

def evaluate(model, encoders, loader, device):
    model.eval()
    true, pred, probs = [], [], []
    with torch.no_grad():
        for Xlist, yb in loader:
            yb = yb.to(device)
            zs = torch.stack([
                encoders[mod](x.to(device)) 
                for mod, x in zip(encoders.keys(), Xlist)
            ], dim=1)
            
            logits = model(zs)
            batch_probs = torch.softmax(logits, dim=1)
            batch_preds = logits.argmax(1)
            
            true.append(yb.cpu())
            pred.append(batch_preds.cpu())
            probs.append(batch_probs[:, 1].cpu())  # For binary classification
    
    true = torch.cat(true).numpy()
    pred = torch.cat(pred).numpy()
    probs = torch.cat(probs).numpy()
    
    acc = (true == pred).mean() * 100
    f1 = f1_score(true, pred, average="macro")
    auc = roc_auc_score(true, probs)
    return acc, f1, auc

def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load encoders
    encoders = {mod: load_encoder(mod, args.checkpoint_dir, args.latent_dim).to(args.device) 
               for mod in args.modalities}
    if not args.finetune:
        for encoder in encoders.values():
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
    
    # Prepare cross-validation
    data = np.load(args.data_file)
    y = data['y']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"\nFold {fold_idx}/10")
        
        # Prepare data
        train_ds = MultiModalDataset(args.data_file, args.modalities, train_idx)
        val_ds = MultiModalDataset(args.data_file, args.modalities, val_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = MLPFusion(
            dim_latent=args.latent_dim,
            n_modalities=len(args.modalities),
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(args.device)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=args.lr/10)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0
        best_model_path = os.path.join(args.output_dir, f"fold{fold_idx}_best.pth")
        for epoch in range(1, args.epochs+1):
            train_acc = train_epoch(model, encoders, train_loader, optimizer, criterion, 
                                  args.device, args.l1_lambda)
            val_acc, val_f1, val_auc = evaluate(model, encoders, val_loader, args.device)
            scheduler.step()
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {epoch} with Val F1: {val_f1:.4f}")
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        # Load best model for this fold and evaluate
        model.load_state_dict(torch.load(best_model_path,weights_only=True))
        val_acc, val_f1, val_auc = evaluate(model, encoders, val_loader, args.device)
        fold_results.append({'f1': val_f1, 'auc': val_auc, 'model_path': best_model_path})
    
    # Print final results
    f1_scores = [res['f1'] for res in fold_results]
    auc_scores = [res['auc'] for res in fold_results]
    print(f"\nAverage F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    # Evaluate on independent test set
    if args.test_file:
        print("\nEvaluating on independent test set...")
        test_ds = MultiModalDataset(args.test_file, args.modalities)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate each fold's model on test set
        test_results = []
        for fold_res in fold_results:
            model = MLPFusion(
                dim_latent=args.latent_dim,
                n_modalities=len(args.modalities),
                num_classes=args.num_classes,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            ).to(args.device)
            model.load_state_dict(torch.load(fold_res['model_path'],weights_only=True))
            
            test_acc, test_f1, test_auc = evaluate(model, encoders, test_loader, args.device)
            test_results.append({
                'fold': fold_results.index(fold_res) + 1,
                'acc': test_acc,
                'f1': test_f1,
                'auc': test_auc
            })
        
        # Print test results
        print("\nTest Set Results:")
        for res in test_results:
            print(f"Fold {res['fold']}: Acc={res['acc']:.1f}%, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
        
        avg_test_f1 = np.mean([res['f1'] for res in test_results])
        avg_test_auc = np.mean([res['auc'] for res in test_results])
        print(f"\nAverage Test F1: {avg_test_f1:.4f}")
        print(f"Average Test AUC: {avg_test_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", required=True, 
                       help="List of modality names to use")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to training data .npz file")
    parser.add_argument("--test_file", type=str, default=None,
                       help="Path to independent test data .npz file")
    parser.add_argument("--checkpoint_dir", type=str, default="./pretrained",
                       help="Directory containing pretrained encoders")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save results and models")
    parser.add_argument("--latent_dim", type=int, default=256,
                       help="Dimension of latent space")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for MLP")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of output classes")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                       help="Weight decay for optimizer")
    parser.add_argument("--l1_lambda", type=float, default=1e-4,
                       help="L1 regularization strength")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--finetune", action="store_true",
                       help="Fine-tune encoder parameters")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)