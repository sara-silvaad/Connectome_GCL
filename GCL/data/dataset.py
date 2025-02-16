import os
import torch
import numpy as np
import pandas as pd
import scipy.io

from torch_geometric.data import InMemoryDataset, Data
import hashlib


class Dataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset_name,
        x_type="one_hot",
        matrix_file="../data/original_data/scs_desikan.mat",
        negs=False,
        num_nodes=87,
        transform=None,
        pre_transform=None,
        fc_paths=None,
        labels_path="../data/original_data/HCP_behavioral.csv",
    ):
        self.matrix_file = matrix_file
        self.x_type = x_type
        self.dataset_name = dataset_name
        self.negs = negs
        self.num_nodes = num_nodes
        self.fc_paths = fc_paths
        self.labels_path = labels_path

        self.unique_id = self._generate_unique_id(
            dataset_name, matrix_file, negs, num_nodes, x_type
        )
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.matrix_file]

    @property
    def processed_file_names(self):
        return [f"data_{self.unique_id}.pt"]

    def _generate_unique_id(self, dataset_name, matrix_file, negs, num_nodes, x_type):
        params_str = f"{dataset_name}-{matrix_file}-{negs}-{num_nodes}-{x_type}-GCL"
        return hashlib.md5(params_str.encode()).hexdigest()

    def download(self):
        pass  # Implement if necessary

    def process(self):
        label_data = self._load_label_data()
        mat = scipy.io.loadmat(self.raw_paths[0])
        matrix = self._get_matrix(mat)
        subject_list = mat["subject_list"].squeeze()
        data_list = [
            self._process_subject(subject_name, matrix[:, :, idx], label_data)
            for idx, subject_name in enumerate(subject_list)
            if subject_name in label_data
        ]

        self.data, self.slices = self.collate([d for d in data_list if d is not None])
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _load_label_data(self):
        label_data = pd.read_csv(self.labels_path)

        if self.dataset_name == "Gender":
            return dict(zip(label_data["Subject"], label_data["Gender"]))

        elif self.dataset_name == "Age":
            return dict(zip(label_data["Subject"], label_data["Age"]))

        elif self.dataset_name == "Alcohol":
            valid_labels = {0, 1, 6, 7}

            filtered_data = label_data[
                label_data["SSAGA_Alc_Hvy_Max_Drinks"].isin(valid_labels)
            ]

            return dict(
                zip(filtered_data["Subject"], filtered_data["SSAGA_Alc_Hvy_Max_Drinks"])
            )

    def _get_matrix(self, mat):
        chosen_matrix_name = (
            "scs" if self.matrix_file.endswith("scs_desikan.mat") else "fcs"
        )
        return mat.get(chosen_matrix_name, None)

    def _process_subject(self, subject_name, subject_matrix, label_data):

        print(f"procesing subject {subject_name}")
        try:
            # sc
            sc = self._get_sc(subject_matrix)
            edge_index_sc, edge_weight_sc = self._convert_to_edge_index(sc)

            # fc
            fc = self._get_fc(subject_name)
            edge_index_fc, edge_weight_fc = self._convert_to_edge_index(fc)

            # features
            x = self._get_x()
            x_fc = self._get_x()

            # labels
            y = self._get_label(subject_name, label_data)

            if y is not None:
                return Data(
                    x=x,
                    x_fc=x_fc,
                    fc=fc,
                    sc=sc,
                    edge_index_fc=edge_index_fc,
                    edge_weight_fc=edge_weight_fc,
                    edge_index_sc=edge_index_sc,
                    edge_weight_sc=edge_weight_sc,
                    y=y,
                )

        except UnboundLocalError as e:
            print(f"Error processing subject {subject_name}: {e}")

        return None

    def _get_fc(self, subject_name):

        path_to_subject_fc = os.path.join(f"{self.fc_paths}/{subject_name}")
        fc_list = [
            np.load(os.path.join(path_to_subject_fc, fc))
            for fc in os.listdir(path_to_subject_fc)
            if ("corr" in fc)
        ]

        fc_list_processed = []
        for fc in fc_list:
            fc = fc[:, :].astype(np.float32)

            if self.num_nodes == 68:
                fc = fc[19:, 19:].astype(np.float32)

            if self.negs == False:
                fc = np.where(fc < 0, 0, fc)

            fc_list_processed.append(fc)

        if fc_list_processed != []:
            fc_new = np.array(fc_list_processed)

        return torch.tensor(fc_new, dtype=torch.float32).view(-1, self.num_nodes)

    def _get_sc(self, M):

        if self.num_nodes == 87:
            M = M[:, :].astype(np.float32)
        else:
            M = M[19:, 19:].astype(np.float32)

        M = M + M.T - np.diag(M.diagonal())

        return torch.tensor(M, dtype=torch.float32)

    def _convert_to_edge_index(self, M):

        edge_index = M.nonzero(as_tuple=False)
        edge_index = edge_index.t().contiguous()
        edge_weights = M[edge_index[0], edge_index[1]].view(-1, 1)

        return edge_index, edge_weights

    def _get_label(self, subject_name, label_data):
        label = label_data[subject_name]

        if self.dataset_name == "Gender":
            t = torch.tensor([0 if label == "M" else 1], dtype=torch.int64)

        elif self.dataset_name == "Age":
            age_mapping = {"22-25": 0, "26-30": 1, "31-35": 2}
            t = (
                torch.tensor([age_mapping.get(label, 3)], dtype=torch.int64)
                if label in age_mapping
                else None
            )

        elif self.dataset_name == "Alcohol":
            alcohol_mapping = {0: 0, 1: 0, 6: 1, 7: 1}
            if label in alcohol_mapping:
                t = torch.tensor([alcohol_mapping[label]], dtype=torch.int64)
            else:
                t = None
        else:
            raise ValueError("Unsupported dataset_name")

        if t is None:
            return None
        else:
            return t.to(torch.float32)

    def _get_x(self):

        if self.x_type == "one_hot":
            return torch.eye(self.num_nodes, dtype=torch.float32)

        raise ValueError("Unsupported x_type")
