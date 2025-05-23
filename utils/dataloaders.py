import math
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dgl
import torch
import dgl.graphbolt as gb
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    sequences, masks = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(Dataset):
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


class NetFlowDataset:
    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        in_memory=False,
        fraction=None,
        data_type="benign",
        seed=42,
    ):
        graph_dir = os.path.join(data_dir, "graph_data")
        if fraction is not None:
            assert 0 < fraction < 1
            fraction_str = str(fraction).replace(".", "_")
            save_dir = os.path.join(graph_dir, f"{name}_{fraction_str}")
        else:
            save_dir = os.path.join(graph_dir, name)
        if force_reload:
            existing_cache = [False, False, False]
        else:
            existing_cache = self.has_cache(save_dir, data_type)
        if not all(existing_cache):
            self.process(
                name,
                data_dir,
                save_dir,
                in_memory,
                existing_cache,
                fraction,
                data_type,
                seed,
            )
        if data_type == "benign":
            self.train_data = gb.OnDiskDataset(
                os.path.join(save_dir, "train_benign"), include_original_edge_id=True
            ).load()
        elif data_type == "mixed":
            self.train_data = gb.OnDiskDataset(
                os.path.join(save_dir, "train_mixed"), include_original_edge_id=True
            ).load()
        else:
            raise ValueError("data_type must be either 'benign' or 'mixed'")
        self.val_data = gb.OnDiskDataset(
            os.path.join(save_dir, "val"), include_original_edge_id=True
        ).load()
        self.test_data = gb.OnDiskDataset(
            os.path.join(save_dir, "test"), include_original_edge_id=True
        ).load()

    def save_graph(self, graph, save_dir, in_memory):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        labels = graph.edata["Label"]
        node_features = torch.ones([graph.number_of_nodes(), graph.edata["h"].shape[1]])
        edge_features = graph.edata["h"]
        edges = graph.edges(form="uv", order="eid")
        edges = torch.stack(edges, dim=1).T
        np.save(os.path.join(save_dir, "edges.npy"), edges.numpy())
        np.save(os.path.join(save_dir, "node_features.npy"), node_features.numpy())
        np.save(os.path.join(save_dir, "edge_features.npy"), edge_features.numpy())
        np.save(os.path.join(save_dir, "labels.npy"), labels.numpy())
        yaml_content = f"""
dataset_name: homogeneous_graph
graph:
  nodes:
    - num: {graph.number_of_nodes()}
  edges:
    - format: numpy
      path: edges.npy
feature_data:
  - domain: node
    name: h
    format: numpy
    in_memory: {in_memory}
    path: node_features.npy
  - domain: edge
    name: h
    format: numpy
    in_memory: {in_memory}
    path: edge_features.npy
  - domain: edge
    name: seeds
    format: numpy
    in_memory: True
    path: edges.npy
  - domain: edge
    name: labels
    format: numpy
    in_memory: {in_memory}
    path: labels.npy
"""
        metadata_path = os.path.join(save_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            f.write(yaml_content)

    def process(
        self,
        name,
        data_dir,
        save_dir,
        in_memory,
        existing_cache,
        fraction,
        data_type,
        seed,
    ):
        df = pd.read_csv(os.path.join(data_dir, name, f"{name}.csv"))
        if fraction is not None:
            df = df.groupby(by="Attack").sample(frac=fraction, random_state=seed)
        X = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        if "v3" in name:
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
            df, test_size=0.2, random_state=seed, stratify=y["Attack"]
        )
        if data_type == "benign":
            df_train = df_train[df_train["Label"] == 0]
        scaler_path = os.path.join("scalers", f"scaler_{name}.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"Loaded existing scaler from {scaler_path}")
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
                scaler = MinMaxScaler()
            df_train[edge_features] = scaler.transform(df_train[edge_features])
        else:
            scaler = MinMaxScaler()
            df_train[edge_features] = scaler.fit_transform(df_train[edge_features])
            os.makedirs(scaler_path, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)
        df_train["h"] = df_train[edge_features].apply(lambda row: row.tolist(), axis=1)
        df_val_test["h"] = df_val_test[edge_features].apply(
            lambda row: row.tolist(), axis=1
        )
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=seed,
            stratify=df_val_test["Attack"],
        )
        if "v3" in name:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")
        if not existing_cache[0]:
            src_nodes = (
                df_train["IPV4_SRC_ADDR"].astype("category").cat.codes.to_numpy()
            )
            dst_nodes = (
                df_train["IPV4_DST_ADDR"].astype("category").cat.codes.to_numpy()
            )
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
            src_nodes = np.array([node_map[ip] for ip in df_train["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_train["IPV4_DST_ADDR"]])
            train_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_map))
            train_graph.edata["h"] = torch.tensor(df_train["h"].tolist())
            train_graph.edata["Label"] = torch.tensor(df_train["Label"].to_numpy())
            self.save_graph(
                train_graph,
                os.path.join(save_dir, "train_" + data_type),
                in_memory=in_memory,
            )
        if not existing_cache[1]:
            src_nodes = np.array([node_map[ip] for ip in df_val["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_val["IPV4_DST_ADDR"]])
            val_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_map))
            val_graph.edata["h"] = torch.tensor(df_val["h"].tolist())
            val_graph.edata["Label"] = torch.tensor(df_val["Label"].to_numpy())
            self.save_graph(
                val_graph, os.path.join(save_dir, "val"), in_memory=in_memory
            )
        if not existing_cache[2]:
            src_nodes = np.array([node_map[ip] for ip in df_test["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_test["IPV4_DST_ADDR"]])
            test_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_map))
            test_graph.edata["h"] = torch.tensor(df_test["h"].tolist())
            test_graph.edata["Label"] = torch.tensor(df_test["Label"].to_numpy())
            self.save_graph(
                test_graph, os.path.join(save_dir, "test"), in_memory=in_memory
            )

    def __len__(self):
        return 3

    def has_cache(self, save_dir, data_type):
        # check whether there are processed data in graph_dir
        if data_type == "benign":
            train_dir = os.path.join(save_dir, "train_benign")
        else:
            train_dir = os.path.join(save_dir, "train_mixed")
        val_dir = os.path.join(save_dir, "val")
        test_dir = os.path.join(save_dir, "test")
        existing_cache = []
        for path in [train_dir, val_dir, test_dir]:
            if (
                not os.path.exists(path + "/metadata.yaml")
                or not os.path.exists(path + "/edges.npy")
                or not os.path.exists(path + "/node_features.npy")
                or not os.path.exists(path + "/edge_features.npy")
                or not os.path.exists(path + "/labels.npy")
            ):
                existing_cache.append(False)
            else:
                existing_cache.append(True)
        return existing_cache


class GraphDataLoader(gb.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int,
        nhops: int,
        seed: int,
        fanout: int = -1,
        drop_last=False,
        shuffle=False,
        device="cpu",
    ):
        self._drop_last = drop_last
        feature_store = dataset.feature
        graph = dataset.graph
        seeds = dataset.feature.read("edge", None, "seeds").T.type(torch.int32)
        labels = dataset.feature.read("edge", None, "labels").type(torch.int32)
        train_set = gb.ItemSet((seeds, labels), names=("seeds", "labels"))
        self._itemset_length = len(train_set)
        if batch_size == -1:
            self._batch_size = self._itemset_length
        else:
            self._batch_size = batch_size
        datapipe = gb.ItemSampler(
            train_set,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        datapipe = datapipe.sample_neighbor(
            graph, [fanout] * nhops
        )  # Sample all neighbors of layers up to nhops
        datapipe = datapipe.fetch_feature(
            feature_store, node_feature_keys=["h"], edge_feature_keys=["h"]
        )
        datapipe = datapipe.copy_to(device)
        super(GraphDataLoader, self).__init__(datapipe)

    def __len__(self):
        if self._drop_last:
            return self._itemset_length // self._batch_size
        return math.ceil(self._itemset_length / self._batch_size)
