import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from utils.shift_segments import (
    MS_PER_DAY,
    filter_df_by_segment,
    get_shift_aware_day_split,
    resolve_segment,
)

torch.serialization.add_safe_globals(
    [
        DataEdgeAttr,
        DataTensorAttr,
        GlobalStorage,
    ]
)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    sequences, masks = zip(*batch, strict=False)
    sequences_padded = pad_sequence(list(sequences), batch_first=True, padding_value=0)
    masks_padded = pad_sequence(list(masks), batch_first=True, padding_value=0)
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


class NetFlowDataset:
    VALID_SPLIT_MODES = {"stratified", "temporal", "temporal_shift_aware"}
    VALID_DISTRIBUTION_SEGMENTS = {"pre_shift", "post_shift"}
    TIMESTAMP_COL = "FLOW_START_MILLISECONDS"

    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        fraction=None,
        data_type="benign",
        seed=42,
        split_mode="stratified",
        distribution_segment="pre_shift",
    ):
        self.name = name
        self.data_dir = data_dir
        self.fraction = fraction
        self.data_type = data_type
        self.seed = seed
        self.split_mode = split_mode
        self.distribution_segment = distribution_segment

        if self.split_mode not in self.VALID_SPLIT_MODES:
            allowed = ", ".join(sorted(self.VALID_SPLIT_MODES))
            raise ValueError(f"Unknown split_mode={split_mode!r}. Allowed: {allowed}")
        if self.data_type not in {"benign", "mixed"}:
            raise ValueError(f"Unknown data_type={data_type!r}. Allowed: benign, mixed")
        if self.distribution_segment not in self.VALID_DISTRIBUTION_SEGMENTS:
            allowed = ", ".join(sorted(self.VALID_DISTRIBUTION_SEGMENTS))
            raise ValueError(
                f"Unknown distribution_segment={distribution_segment!r}. "
                f"Allowed: {allowed}"
            )

        # Setup directories
        graph_dir = os.path.join(data_dir, "pyg_graph_data")
        cache_name = self._cache_name()
        self.processed_dir = os.path.join(graph_dir, cache_name)
        self.raw_dir = os.path.join(data_dir, name)
        self.scaler_path = os.path.join(self.processed_dir, "scaler.pkl")

        # Handle force reload
        if force_reload and os.path.exists(self.processed_dir):
            print(
                f"Force reload: Removing existing processed data at {self.processed_dir}"
            )
            shutil.rmtree(self.processed_dir)

        # Check if we need to process
        if self._needs_processing():
            # Check for seed mismatch before processing
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

            self._process()
        else:
            # Check for seed mismatch when loading existing cache
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

        # Load the processed data
        self.train_graph = torch.load(os.path.join(self.processed_dir, "train.pt"))[0]
        self.val_graph = torch.load(os.path.join(self.processed_dir, "val.pt"))[0]
        self.test_graph = torch.load(os.path.join(self.processed_dir, "test.pt"))[0]

    def _cache_name(self):
        if self.fraction is not None:
            assert 0 < self.fraction < 1
            fraction_str = str(self.fraction).replace(".", "_")
            cache_name = f"{self.name}_{fraction_str}"
        else:
            cache_name = self.name

        if self.split_mode == "stratified":
            return cache_name
        if self.split_mode == "temporal_shift_aware":
            return f"{cache_name}_{self.split_mode}_{self.distribution_segment}"
        return f"{cache_name}_{self.split_mode}"

    def _needs_processing(self):
        """Check if processing is needed"""
        if not os.path.exists(self.processed_dir):
            return True

        required_files = ["train.pt", "val.pt", "test.pt"]
        for filename in required_files:
            if not os.path.exists(os.path.join(self.processed_dir, filename)):
                return True

        return False

    def _ensure_timestamped(self, df):
        if self.TIMESTAMP_COL not in df.columns:
            raise ValueError(
                f"split_mode={self.split_mode!r} requires timestamped v3 data "
                f"with a {self.TIMESTAMP_COL!r} column."
            )

    def _validate_splits(self, splits):
        for split_name, df_split in splits.items():
            if df_split.empty:
                raise ValueError(
                    f"split_mode={self.split_mode!r} produced an empty "
                    f"{split_name} split for dataset={self.name!r}."
                )

    def _split_by_fraction(self, df):
        n_rows = len(df)
        train_end = int(n_rows * 0.8)
        val_end = int(n_rows * 0.9)
        splits = {
            "train": df.iloc[:train_end].copy(),
            "val": df.iloc[train_end:val_end].copy(),
            "test": df.iloc[val_end:].copy(),
        }
        self._validate_splits(splits)
        return splits["train"], splits["val"], splits["test"]

    def _split_by_day_ids(self, df, day_split):
        day_ids = (df[self.TIMESTAMP_COL] // MS_PER_DAY).astype(int)
        train = df.loc[day_ids.isin(day_split.train_day_ids)].copy()
        val = df.loc[day_ids.isin(day_split.val_day_ids)].copy()
        test = df.loc[day_ids.isin(day_split.test_day_ids)].copy()

        if day_split.split_day_id is not None:
            split_day = df.loc[day_ids == day_split.split_day_id].copy()
            split_day = split_day.sort_values(by=self.TIMESTAMP_COL).reset_index(
                drop=True
            )
            split_point = int(len(split_day) * day_split.split_day_val_fraction)
            val = pd.concat([val, split_day.iloc[:split_point]], ignore_index=True)
            test = pd.concat([split_day.iloc[split_point:], test], ignore_index=True)

        splits = {
            "train": train.sort_values(by=self.TIMESTAMP_COL).copy(),
            "val": val.sort_values(by=self.TIMESTAMP_COL).copy(),
            "test": test.sort_values(by=self.TIMESTAMP_COL).copy(),
        }
        self._validate_splits(splits)
        return splits["train"], splits["val"], splits["test"]

    def _split_stratified(self, df):
        df_train, df_val_test = train_test_split(
            df,
            test_size=0.2,
            random_state=self.seed,
            stratify=df["Attack"],
        )
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )
        if self.TIMESTAMP_COL in df.columns:
            df_train = df_train.sort_values(by=self.TIMESTAMP_COL)
            df_val = df_val.sort_values(by=self.TIMESTAMP_COL)
            df_test = df_test.sort_values(by=self.TIMESTAMP_COL)
        return df_train.copy(), df_val.copy(), df_test.copy()

    def _split_temporal(self, df):
        self._ensure_timestamped(df)
        df = df.sort_values(by=self.TIMESTAMP_COL).reset_index(drop=True)
        return self._split_by_fraction(df)

    def _split_temporal_shift_aware(self, df):
        self._ensure_timestamped(df)
        segment = resolve_segment(self.name, self.distribution_segment)
        df = filter_df_by_segment(df, timestamp_col=self.TIMESTAMP_COL, segment=segment)
        df = df.sort_values(by=self.TIMESTAMP_COL).reset_index(drop=True)
        day_split = get_shift_aware_day_split(self.name, segment.name)
        if day_split is not None:
            return self._split_by_day_ids(df, day_split)
        return self._split_by_fraction(df)

    def _split_dataframe(self, df):
        if self.split_mode == "stratified":
            return self._split_stratified(df)
        if self.split_mode == "temporal":
            return self._split_temporal(df)
        if self.split_mode == "temporal_shift_aware":
            return self._split_temporal_shift_aware(df)
        raise ValueError(f"Unsupported split_mode={self.split_mode!r}")

    def _process(self):
        """Process the raw CSV data and create train/val/test splits"""
        print(f"Processing dataset {self.name}...")

        os.makedirs(self.processed_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"))

        if self.fraction is not None:
            df = df.groupby(by="Attack").sample(
                frac=self.fraction, random_state=self.seed
            )

        x = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        if "v3" in self.name:
            edge_features = [
                col
                for col in x.columns
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
                for col in x.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]

        df = pd.concat([x, y], axis=1)

        df_train, df_val, df_test = self._split_dataframe(df)

        if self.data_type == "benign":
            df_train = df_train[df_train["Label"] == 0].copy()

        self._validate_splits({"train": df_train, "val": df_val, "test": df_test})

        scaler = None
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(df_train[edge_features])
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        df_train[edge_features] = scaler.transform(df_train[edge_features])
        df_val[edge_features] = np.clip(
            scaler.transform(df_val[edge_features]), -10, 10
        )
        df_test[edge_features] = np.clip(
            scaler.transform(df_test[edge_features]), -10, 10
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
        num_nodes = len(node_map)

        datasets = {"train": df_train, "val": df_val, "test": df_test}

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

            # Save as list for compatibility
            torch.save([data], os.path.join(self.processed_dir, f"{split_name}.pt"))

        # Save seed information for cache validation
        seed_file = os.path.join(self.processed_dir, ".seed")
        with open(seed_file, "w") as f:
            f.write(str(self.seed))

        print("Done!")

    def __len__(self):
        # Return total number of graphs (for compatibility)
        return 3

    @property
    def num_node_features(self):
        return self.train_graph.x.shape[1]

    @property
    def num_edge_features(self):
        return self.train_graph.edge_attr.shape[1]

    @property
    def num_nodes(self):
        return self.train_graph.num_nodes
