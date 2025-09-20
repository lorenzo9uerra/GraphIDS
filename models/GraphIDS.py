import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SAGELayer(MessagePassing):
    def __init__(self, ndim_in, edim_in, edim_out, agg_type="mean", dropout_rate=0.0):
        super(SAGELayer, self).__init__(aggr=agg_type)
        self.fc_neigh = nn.Linear(edim_in, ndim_in)
        self.fc_edge = nn.Linear(ndim_in * 2, edim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.agg_type = agg_type
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

    def message(self, edge_attr):
        """Define how messages are computed from edge features"""
        return edge_attr

    def forward(self, x, edge_index, edge_attr, edge_couples):
        """
        Args:
            x: Node features [num_nodes, ndim_in]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edim_in]
            edge_couples: Target edge pairs [batch_size, 2]
        """
        node_embeddings = self.propagate(
            edge_index, edge_attr=edge_attr, size=(x.shape[0], x.shape[0])
        )
        node_embeddings = self.relu(self.fc_neigh(node_embeddings))

        src_nodes = edge_couples[:, 0]
        dst_nodes = edge_couples[:, 1]

        edge_embeddings = self.fc_edge(
            torch.cat([node_embeddings[src_nodes], node_embeddings[dst_nodes]], dim=1)
        )
        edge_embeddings = self.dropout(edge_embeddings)
        return node_embeddings, edge_embeddings


class SAGE(nn.Module):
    def __init__(
        self, ndim_in, edim_in, edim_out, nhops, dropout_rate, agg_type="mean"
    ):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()

        if nhops == 1:
            self.layers.append(
                SAGELayer(
                    ndim_in,
                    edim_in,
                    edim_out,
                    agg_type=agg_type,
                    dropout_rate=dropout_rate,
                )
            )
        else:
            self.layers.append(
                SAGELayer(
                    ndim_in,
                    edim_in,
                    edim_in,
                    agg_type=agg_type,
                    dropout_rate=dropout_rate,
                )
            )
            for _ in range(nhops - 2):
                self.layers.append(
                    SAGELayer(
                        ndim_in,
                        edim_in,
                        edim_in,
                        agg_type=agg_type,
                        dropout_rate=dropout_rate,
                    )
                )
            self.layers.append(
                SAGELayer(
                    ndim_in,
                    edim_in,
                    edim_out,
                    agg_type=agg_type,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x, edge_index, edge_attr, edge_couples):
        """
        Args:
            x: Node features [num_nodes, ndim_in]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edim_in]
            edge_couples: Target edge pairs [batch_size, 2]
        """
        for layer in self.layers:
            x, edge_embeddings = layer(x, edge_index, edge_attr, edge_couples)

        return edge_embeddings


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

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
    ):
        super(TransformerAutoencoder, self).__init__()
        if positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoding(
                embed_dim, window_size
            )
        elif positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoding(
                embed_dim, window_size
            )
        else:
            self.positional_encoder = None
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.mask_ratio = mask_ratio
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
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

        if self.positional_encoder is not None:
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
        nhops=1,
        agg_type="mean",
        mask_ratio=0.15,
    ):
        super(GraphIDS, self).__init__()
        self.encoder = SAGE(
            ndim_in, edim_in, edim_out, nhops, dropout, agg_type=agg_type
        )
        self.transformer = TransformerAutoencoder(
            edim_out,
            embed_dim,
            num_heads,
            num_layers,
            ae_dropout,
            window_size,
            positional_encoding,
            mask_ratio,
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
