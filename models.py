import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torchmetrics.classification import Accuracy, AUROC, CohenKappa, ConfusionMatrix, F1Score, Precision, RecallAtFixedPrecision, Recall, PrecisionRecallCurve

from Helper_functions import calculate_metrics, FocalLoss
#%% Model wrapper
class ModelWrapper:
    def __init__(self, model, optimizer, criterion, use_amp=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if use_amp is None:
            enable_amp = torch.cuda.is_available() and not isinstance(criterion, FocalLoss)
        else:
            enable_amp = bool(use_amp)
        self._use_amp = enable_amp
        self._scaler = _make_scaler(enabled=self._use_amp)
        
    def train_step(self, loader):
            self.model.train()
            total_loss = 0
            # Determine device dynamically from the model
            device = next(self.model.parameters()).device

            for batch in loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                
                with _autocast(enabled=self._use_amp):
                    out = self.model(batch)
                    
                    # Slice to get only the target nodes (the first batch_size nodes)
                    # This is equivalent to applying the mask in the full-batch version
                    batch_size = batch.batch_size
                    out_sliced = out[:batch_size]
                    y_sliced = batch.y[:batch_size]
                    
                    loss = self.criterion(out_sliced, y_sliced)
                
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN, stopping training")

                # Accumulate loss
                total_loss += float(loss.detach())

                if self._use_amp:
                    self._scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            # Return the average loss over all batches
            return total_loss / len(loader)
 # In models.py

    # ... inside ModelWrapper class ...

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        # Determine device dynamically
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with _autocast(enabled=self._use_amp):
                    out = self.model(batch)
                    
                    # Slice to get only the target nodes (first batch_size nodes)
                    batch_size = batch.batch_size
                    out_sliced = out[:batch_size]
                    y_sliced = batch.y[:batch_size]
                    
                    loss = self.criterion(out_sliced, y_sliced)
                    total_loss += float(loss.detach())
                
                # Collect predictions for metrics
                probs = torch.nn.functional.softmax(out_sliced, dim=1)
                pred = out_sliced.argmax(dim=1)
                
                # Move to CPU to save GPU memory during accumulation
                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(y_sliced.cpu())

        # Concatenate all mini-batches to calculate global metrics
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        return total_loss / len(loader), metrics
    
try:
    from torch.amp import autocast as _torch_autocast
    from torch.amp import GradScaler as _TorchGradScaler

    def _autocast(enabled: bool):
        return _torch_autocast("cuda", enabled=enabled)

    def _make_scaler(enabled: bool):
        return _TorchGradScaler("cuda", enabled=enabled)
except (ImportError, TypeError, AttributeError):
    from torch.cuda.amp import autocast as _torch_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore

    def _autocast(enabled: bool):
        return _torch_autocast(enabled=enabled)

    def _make_scaler(enabled: bool):
        return _TorchGradScaler(enabled=enabled)


#%% models

class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network model.
    """
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output raw logits
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, num_heads):
        super(GAT, self).__init__()
        # Keep the total latent size roughly equal to hidden_units while limiting per-head width
        per_head_dim = max(1, math.ceil(hidden_units / num_heads))
        total_hidden = per_head_dim * num_heads
        self.conv1 = GATConv(num_node_features, per_head_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(total_hidden, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
    
class GIN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x = data.x  # only use node features, no graph structure
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x