import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class SAGELayer(MessagePassing):
    def __init__(self, ndim_in, edim_in, edim_out, agg_type="mean", dropout_rate=0.0):
        super().__init__(aggr=agg_type)
        self.fc_neigh = nn.Linear(edim_in, ndim_in)
        self.fc_edge = nn.Linear(ndim_in * 2, edim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.agg_type: str = agg_type
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

    def message(self, edge_attr):  # type: ignore[override]
        """
        Copy edge features as messages.
        DGL equivalent: fn.copy_e("h", "m")
        """
        return edge_attr

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate edge features to destination nodes using mean aggregation.
        DGL equivalent: fn.mean("m", "h_neigh")
        """
        return scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean"
        )

    def forward(self, edge_index, edge_attr, edge_couples, num_nodes):
        """
        Aggregate edge features to nodes, then compute edge embeddings.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edim_in] - the actual input data
            edge_couples: Target edge pairs [batch_size, 2] - edges to embed
            num_nodes: Number of nodes in the graph

        Returns:
            edge_embeddings: [batch_size, edim_out] - final edge representations

        Process:
            1. Aggregate edge features to nodes (mean of incoming edges)
            2. Transform aggregated features: fc_neigh + ReLU
            3. For target edges: concat source & dest node embeddings
            4. Project to final edge embeddings: fc_edge + dropout
        """
        # Aggregate edge features to nodes (DGL: fn.copy_e + fn.mean)
        node_embeddings = self.propagate(  # type: ignore
            edge_index, edge_attr=edge_attr, size=(num_nodes, num_nodes)
        )
        # Transform aggregated features and activate
        node_embeddings = self.relu(self.fc_neigh(node_embeddings))

        # Compute edge embeddings from concatenated source and destination node embeddings
        edge_embeddings = self.fc_edge(
            torch.cat(
                [
                    node_embeddings[edge_couples[:, 0]],
                    node_embeddings[edge_couples[:, 1]],
                ],
                dim=1,
            )
        )
        return self.dropout(edge_embeddings)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        window_size,
        positional_encoding,
        mask_ratio,
        feedforward_dim=256,
    ):
        super().__init__()
        self.feedforward_dim = feedforward_dim
        if positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoding(
                embed_dim, window_size
            )
        elif positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoding(
                embed_dim, window_size
            )
        else:
            self.positional_encoder = nn.Identity()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.mask_ratio = mask_ratio
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embed_dim, input_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        for name, param in self.encoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.decoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, src, padding_mask=None):
        src = self.input_projection(src)

        # Identity function if no positional encoding is used
        src = self.positional_encoder(src)

        src_key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.bool()
            src_key_padding_mask = ~torch.any(padding_mask, dim=-1)

        if self.training and self.mask_ratio > 0:
            seq_len = src.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool),
                diagonal=1,
            )
            mask = mask & (
                torch.rand(seq_len, seq_len, device=src.device) < self.mask_ratio
            )
            attention_mask = mask | mask.T
        else:
            attention_mask = None

        memory = self.encoder(
            src, mask=attention_mask, src_key_padding_mask=src_key_padding_mask
        )

        output = self.decoder(
            src,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=src_key_padding_mask,
        )

        output = self.output_projection(output)
        return output


class GraphIDS(nn.Module):
    def __init__(
        self,
        ndim_in,
        edim_in,
        edim_out,
        embed_dim,
        num_heads,
        num_layers,
        window_size=512,
        dropout=0.0,
        ae_dropout=0.1,
        positional_encoding=None,
        agg_type="mean",
        mask_ratio=0.15,
        ae_feedforward_dim=256,
    ):
        super().__init__()
        self.encoder = SAGELayer(ndim_in, edim_in, edim_out, agg_type, dropout)
        self.transformer = TransformerAutoencoder(
            edim_out,
            embed_dim,
            num_heads,
            num_layers,
            ae_dropout,
            window_size,
            positional_encoding,
            mask_ratio,
            ae_feedforward_dim,
        )

    def save_checkpoint(self, path, optimizer=None, epoch=0, threshold=None):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["threshold"]
