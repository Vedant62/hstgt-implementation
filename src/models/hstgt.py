import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData

from .temporal_encoder import TemporalEncoder


class HSTGTBlock(nn.Module):
    """
    One interleaved block: Temporal Self-Attention → Spatial HGT Attention.
    This allows cross-stock information to flow after each temporal update.
    """

    def __init__(
        self,
        d_model: int,
        nhead_temporal: int,
        nhead_spatial: int,
        metadata: tuple,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Temporal update (per-node Transformer layer) ──────────────────────
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_temporal,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # ── Spatial update (HGT over heterogeneous graph) ─────────────────────
        self.spatial_conv = HGTConv(
            in_channels  = d_model,
            out_channels = d_model,
            metadata     = metadata,   
            heads        = nhead_spatial,
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],   # {'stock': (N, d_model)}
        edge_index_dict: dict,             # from HeteroData
    ) -> dict[str, torch.Tensor]:

        # ── Temporal self-attention (treat each node as a seq of length 1) ────
        node_feats = x_dict["stock"]                         # (N, d_model)
        node_feats = node_feats.unsqueeze(1)                 # (N, 1, d_model)
        node_feats = self.temporal_layer(node_feats)
        node_feats = node_feats.squeeze(1)                   # (N, d_model)
        node_feats = self.temporal_norm(node_feats)

        x_dict = {"stock": node_feats}

        # ── Spatial HGT attention across the multiplex graph ──────────────────
        out_dict = self.spatial_conv(x_dict, edge_index_dict)
        # HGTConv can return None for types with no edges; handle gracefully
        for key in out_dict:
            if out_dict[key] is None:
                out_dict[key] = x_dict.get(key, torch.zeros_like(node_feats))

        # Residual connection + norm
        out_dict["stock"] = self.spatial_norm(
            out_dict["stock"] + self.dropout(node_feats)
        )
        return out_dict


class HSTGT(nn.Module):
    """
    Full Heterogeneous Spatio-Temporal Graph Transformer.

    Architecture:
      1. TemporalEncoder    : raw time series → initial node embedding
      2. N x HSTGTBlock     : interleaved temporal + spatial attention
      3. Output MLP         : embedding → predicted next-day log return
    """

    def __init__(
        self,
        input_dim: int  = 8,    # raw feature dimension per timestep
        d_model: int    = 64,   # hidden dimension throughout the model
        seq_len: int    = 30,   # input sequence length
        nhead_temporal: int = 4,
        nhead_spatial:  int = 4,
        num_blocks:     int = 2,  # Number of HSTGT stacked blocks
        metadata:       tuple = None,
        dropout:        float = 0.1,
    ):
        super().__init__()

        if metadata is None:
            # Default metadata when building before seeing real data
            node_types = ["stock"]
            edge_types = [
                ("stock", "mathematical", "stock"),
                ("stock", "structural",   "stock"),
                ("stock", "semantic",     "stock"),
            ]
            metadata = (node_types, edge_types)

        # Stage 1: Temporal encoding (shared across all stocks)
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead_temporal,
            num_layers=2,
            seq_len=seq_len,
            dropout=dropout,
        )

        # Stage 2: Stacked HSTGT blocks
        self.blocks = nn.ModuleList([
            HSTGTBlock(
                d_model=d_model,
                nhead_temporal=nhead_temporal,
                nhead_spatial=nhead_spatial,
                metadata=metadata,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Stage 3: Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        data: HeteroData with
            data['stock'].x  → (N, T, F)
            data['stock', edge_type, 'stock'].edge_index
        Returns: predictions of shape (N,)
        """
        x = data["stock"].x  # (N, T, F)

        node_embeddings = self.temporal_encoder(x)  # (N, d_model)
        x_dict = {"stock": node_embeddings}

        edge_index_dict = {}
        for store in data.edge_stores:
            if hasattr(store, "_key") and store._key is not None:
                edge_type = store._key
                if hasattr(store, "edge_index") and store.edge_index.shape[1] > 0:
                    edge_index_dict[edge_type] = store.edge_index

        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict)

        preds = self.output_head(x_dict["stock"]).squeeze(-1)  # (N,)
        return preds