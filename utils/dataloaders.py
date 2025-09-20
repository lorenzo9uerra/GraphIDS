import os
import pickle
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ]
)


def collate_fn(batch):
    sequences, masks = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(TorchDataset):
    def __init__(self, data, window, device, step=None):
        self.data = data
        self.window = window
        self.device = device
        if step is None:
            self.step = window
        else:
            self.step = step

    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = min(start_idx + self.window, len(self.data))
        x = self.data[start_idx:end_idx].to(self.device)
        mask = torch.ones_like(x, dtype=torch.bool).to(self.device)
        return x, mask

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.step + 1)


class PyGDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        data_dir,
        force_reload,
        fraction=None,
        data_type="benign",
        seed=42,
        split="train",
    ):
        self.name = name
        self.data_dir = data_dir
        self.fraction = fraction
        self.data_type = data_type
        self.seed = seed
        self.split = split

        graph_dir = os.path.join(data_dir, "pyg_graph_data")
        if fraction is not None:
            assert 0 < fraction < 1
            fraction_str = str(fraction).replace(".", "_")
            self.processed_dir_path = os.path.join(graph_dir, f"{name}_{fraction_str}")
        else:
            self.processed_dir_path = os.path.join(graph_dir, name)

        if force_reload and os.path.exists(self.processed_dir_path):
            print(
                f"Force reload: Removing existing processed data at {self.processed_dir_path}"
            )
            shutil.rmtree(self.processed_dir_path)

        super().__init__(root=self.processed_dir_path)

        if split == "train":
            self.data_list = self.load(self.processed_paths[0])
        elif split == "val":
            self.data_list = self.load(self.processed_paths[1])
        elif split == "test":
            self.data_list = self.load(self.processed_paths[2])
        else:
            raise ValueError("split must be one of 'train', 'val', 'test'")

    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]

    @property
    def processed_file_names(self):
        files = ["train.pt", "val.pt", "test.pt"]
        return files

    @property
    def raw_dir(self):
        return os.path.join(self.data_dir, self.name)

    @property
    def processed_dir(self):
        return self.processed_dir_path

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"))

        if self.fraction is not None:
            df = df.groupby(by="Attack").sample(
                frac=self.fraction, random_state=self.seed
            )

        X = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        if "v3" in self.name:
            edge_features = [
                col
                for col in X.columns
                if col
                not in [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                ]
            ]
        else:
            edge_features = [
                col
                for col in X.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]

        df = pd.concat([X, y], axis=1)

        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=y["Attack"]
        )

        if self.data_type == "benign":
            df_train = df_train[df_train["Label"] == 0]

        scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"Loaded existing scaler from {scaler_path}")
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
                scaler = MinMaxScaler()
                scaler.fit(df_train[edge_features])
        else:
            scaler = MinMaxScaler()
            scaler.fit(df_train[edge_features])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        df_train[edge_features] = scaler.transform(df_train[edge_features])
        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )

        if "v3" in self.name:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")

        unique_nodes = pd.concat(
            [
                df_train["IPV4_SRC_ADDR"],
                df_train["IPV4_DST_ADDR"],
                df_val["IPV4_SRC_ADDR"],
                df_val["IPV4_DST_ADDR"],
                df_test["IPV4_SRC_ADDR"],
                df_test["IPV4_DST_ADDR"],
            ]
        ).unique()
        node_map = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(node_map)

        datasets = {"train": df_train, "val": df_val, "test": df_test}
        processed_data = {}

        for split_name, df_split in datasets.items():
            src_nodes = np.array([node_map[ip] for ip in df_split["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_split["IPV4_DST_ADDR"]])
            edge_index = torch.tensor(
                np.array([src_nodes, dst_nodes]), dtype=torch.long
            )
            edge_attr = torch.tensor(df_split[edge_features].values, dtype=torch.float)
            edge_labels = torch.tensor(df_split["Label"].values, dtype=torch.long)
            x = torch.ones(num_nodes, edge_attr.shape[1], dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_labels=edge_labels,
                num_nodes=num_nodes,
            )
            # List format for compatibility
            processed_data[split_name] = [data]

        self.save(processed_data["train"], self.processed_paths[0])
        self.save(processed_data["val"], self.processed_paths[1])
        self.save(processed_data["test"], self.processed_paths[2])


class NetFlowDataset:
    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        fraction=None,
        data_type="benign",
        seed=42,
    ):
        self.train_data = PyGDataset(
            name, data_dir, force_reload, fraction, data_type, seed, split="train"
        )
        self.val_data = PyGDataset(
            name, data_dir, force_reload, fraction, data_type, seed, split="val"
        )
        self.test_data = PyGDataset(
            name, data_dir, force_reload, fraction, data_type, seed, split="test"
        )

    def __len__(self):
        return len(self.train_data) + len(self.val_data) + len(self.test_data)

    @property
    def train_graph(self):
        return self.train_data[0]

    @property
    def val_graph(self):
        return self.val_data[0]

    @property
    def test_graph(self):
        return self.test_data[0]

    @property
    def num_node_features(self):
        return self.train_graph.x.shape[1]

    @property
    def num_edge_features(self):
        return self.train_graph.edge_attr.shape[1]

    @property
    def num_nodes(self):
        return self.train_graph.num_nodes
