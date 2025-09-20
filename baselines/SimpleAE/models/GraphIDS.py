import torch
import torch.nn as nn
import dgl.function as fn


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim_in, edim_out, agg_type="mean", dropout_rate=0.0):
        super(SAGELayer, self).__init__()
        self.fc_neigh = nn.Linear(edim_in, ndim_in)
        self.fc_edge = nn.Linear(ndim_in * 2, edim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.agg_type = agg_type
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_neigh.weight, gain=gain)

    def forward(self, block, nfeats, efeats, seeds):
        # Local scope to avoid in-place modification of node and edge features
        with block.local_scope():
            block.srcdata["h"] = nfeats
            block.dstdata["h"] = nfeats[: block.number_of_dst_nodes()]
            block.edata["h"] = efeats
            if self.agg_type == "mean":
                block.update_all(fn.copy_e("h", "m"), fn.mean("m", "h_neigh"))
                block.dstdata["h"] = self.relu(self.fc_neigh(block.dstdata["h_neigh"]))
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(self.agg_type)
                )

            # Compute edge embeddings
            u, v = seeds
            edge = self.fc_edge(
                torch.cat([block.dstdata["h"][u], block.dstdata["h"][v]], 1)
            )
            edge = self.dropout(edge)
            return block.dstdata["h"], edge


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
            if nhops > 2:  # Add more layers if nhops > 2
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

    def forward(self, block, nfeats, efeats, seeds=None):
        if seeds == None:  # If full graph is used instead of a block
            seeds = block.edges()
        for layer in self.layers:
            nfeats, e_embeddings = layer(block, nfeats, efeats, seeds)

        return e_embeddings


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


class SimpleAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 24)
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return reconstructed


class GraphIDS(nn.Module):
    def __init__(
        self,
        ndim_in,
        edim_in,
        edim_out,
        nhops=1,
        dropout=0.0,
        agg_type="mean",
    ):
        super(GraphIDS, self).__init__()
        self.encoder = SAGE(
            ndim_in, edim_in, edim_out, nhops, dropout, agg_type=agg_type
        )
        self.autoencoder = SimpleAE(edim_out)

    def forward(self, block, nfeats, efeats, seeds=None):
        edge_embeddings = self.encoder(block, nfeats, efeats, seeds)
        reconstructed = self.autoencoder(edge_embeddings)
        return edge_embeddings, reconstructed

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
