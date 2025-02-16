import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from abc import ABC, abstractmethod


class GCNEncoderDecoderClassifier(torch.nn.Module, ABC):
    def __init__(
        self,
        hidden_dims,
        num_classes,
        pooling_type="ave",
        num_nodes=87,
        concatenate=True,
        prob_of_diappearance=0.2,
        max_substraction=87,
        negs=False,
    ):
        super(GCNEncoderDecoderClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.concatenate = concatenate
        self.prob_of_diappearance = prob_of_diappearance
        self.max_substraction = max_substraction
        self.convs = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.negs = negs

        for i in range(len(hidden_dims) - 1):
            conv = GCNConv(hidden_dims[i], hidden_dims[i + 1])
            self.convs.append(conv)

    def encoder(self, x, edge_index, edge_weights):
        layer_output = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weights))
            layer_output.append(x)

        if self.concatenate and len(self.convs) > 1:
            node_emb = torch.cat(layer_output, dim=-1)
        else:
            node_emb = layer_output[-1]
        return node_emb

    def graph_embedding(self, node_emb, batch):
        if self.pooling_type == "ave":
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling_type == "sum":
            graph_emb = torch.sum(node_emb, dim=1).unsqueeze(-1)
        elif self.pooling_type == "concatenate":
            graph_emb = node_emb.view(-1, self.num_nodes * sum(self.hidden_dims[1:]))
        elif self.pooling_type == "ave_on_nodes":
            graph_emb = torch.mean(node_emb, dim=1).view(-1, self.num_nodes)
        else:
            graph_emb = global_max_pool(node_emb, batch)
        return graph_emb

    def decoder(self, emb):

        if self.concatenate:
            node_emb = emb.view(-1, self.num_nodes, sum(self.hidden_dims[1:]))
        else:
            node_emb = emb.view(-1, self.num_nodes, self.hidden_dims[-1])
        node_emb_transpose = node_emb.transpose(1, 2)

        if self.negs:
            reconstructed_adj = torch.tanh(torch.matmul(node_emb, node_emb_transpose))
        else:
            reconstructed_adj = torch.relu(torch.matmul(node_emb, node_emb_transpose))

        reconstructed_adj = reconstructed_adj.view(-1, self.num_nodes)

        return reconstructed_adj

    @abstractmethod
    def forward(self, data):
        pass

    def _substract_random_ones(self, x, edge_index, edge_weights, max):
        x = x.view(-1, self.num_nodes, self.num_nodes).to("cuda")
        matrices = torch.zeros(x.shape).to("cuda")
        for i in range(x.shape[0]):
            num_neg_ones = torch.randint(0, max - 1, (1,)).item()

            if num_neg_ones > 0:
                # Select unique random positions along the diagonal
                indices = torch.randperm(self.num_nodes)[:num_neg_ones]

                # Set these positions to -1
                matrices[i, indices, indices] = -1
        return (x + matrices).view(-1, self.num_nodes), edge_index, edge_weights

    def _remove_edges(self, x, edge_index, edge_weights, p):
        # Generate a random mask with the same length as edge_index.shape[1]
        mask = torch.rand(edge_index.shape[1]) > p

        # Apply the mask to filter the edges and their corresponding weights
        edge_index_r = edge_index[:, mask]
        edge_weights_r = edge_weights[mask]

        return x, edge_index_r, edge_weights_r

    def _get_region(self, x__, edge_index, edge_weights, region):
        x = x__.clone()
        # Define the range of nodes based on the region
        if region == "subcortical":
            start, end = 0, 19
        elif region == "right":
            start, end = 19, 54
        elif region == "left":
            start, end = 54, 87
        else:
            raise ValueError("Incorrect region")

        # Vectorize the operation to calculate the mask
        num_nodes = edge_index.max().item() + 1
        m = 87

        # Create the tensor of interval indices
        interval_indices = torch.arange(0, num_nodes, m).view(-1, 1) + torch.arange(
            start, end
        ).view(1, -1)

        # Create the boolean mask in a vectorized manner
        interval_set = set(interval_indices.view(-1).tolist())
        mask = torch.tensor(
            [
                (i.item() not in interval_set and j.item() not in interval_set)
                for i, j in zip(edge_index[0], edge_index[1])
            ],
            dtype=torch.bool,
        )

        # Filter the pairs and edge_weights that do not meet the condition
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_weights = edge_weights[mask]

        # Normalize the indices in a vectorized manner
        def normalize_index(x, start, end, m=87):
            i = x // m
            return (x - start + (i * (end - start))) % m

        normalized_edge_index = torch.vstack(
            [
                normalize_index(filtered_edge_index[0], start, end),
                normalize_index(filtered_edge_index[1], start, end),
            ]
        )

        x_ = x.view(-1, 87, 87)
        x_[:, :, :start] = 0
        x_[:, :, end:] = 0
        x_region = x_.view(-1, 87)

        return x_region, normalized_edge_index, filtered_edge_weights


class GCL_region(GCNEncoderDecoderClassifier):
    def forward(self, data):
        # sample minibatch of examples x1,...xN
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index_sc,
            data.edge_weight_sc,
            data.batch,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            x, edge_index, edge_weights, batch = (
                x.to(device),
                edge_index.to(device),
                edge_weights.to(device),
                batch.to(device),
            )
        else:
            device = torch.device("cpu")

        # augment each example twice to get x1^, ..., xN^, x1',...,xN'
        regions = ["subcortical", "right", "left"]
        x_1, edge_index_1, edge_weights_1 = self._get_region(
            x, edge_index, edge_weights, regions[0]
        )
        x_2, edge_index_2, edge_weights_2 = self._get_region(
            x, edge_index, edge_weights, regions[1]
        )
        x_3, edge_index_3, edge_weights_3 = self._get_region(
            x, edge_index, edge_weights, regions[2]
        )

        # embed example with encoder f to get z1^,...,zN^, z1',..., zN'
        node_emb = self.encoder(x, edge_index, edge_weights)
        node_emb_1 = self.encoder(x_1, edge_index_1, edge_weights_1)
        node_emb_2 = self.encoder(x_2, edge_index_2, edge_weights_2)
        node_emb_3 = self.encoder(x_3, edge_index_3, edge_weights_3)

        graph_emb = self.graph_embedding(node_emb, batch)
        graph_emb_1 = self.graph_embedding(node_emb_1, batch)
        graph_emb_2 = self.graph_embedding(node_emb_2, batch)
        graph_emb_3 = self.graph_embedding(node_emb_3, batch)

        return graph_emb, graph_emb_1, graph_emb_2, graph_emb_3


class GCL_FeatureMaskingEdgeDropping(GCNEncoderDecoderClassifier):
    def forward(self, data):
        # sample minibatch of examples x1,...xN
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index_sc,
            data.edge_weight_sc,
            data.batch,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            x, edge_index, edge_weights, batch = (
                x.to(device),
                edge_index.to(device),
                edge_weights.to(device),
                batch.to(device),
            )
        else:
            device = torch.device("cpu")

        # augment each example twice to get x1^, ..., xN^, x1',...,xN'
        x_1, edge_index_1, edge_weights_1 = self._remove_edges(
            x, edge_index, edge_weights, self.prob_of_diappearance
        )
        x_2, edge_index_2, edge_weights_2 = self._substract_random_ones(
            x, edge_index, edge_weights, self.max_substraction
        )

        # embed example with encoder f to get z1^,...,zN^, z1',..., zN'
        node_emb = self.encoder(x, edge_index, edge_weights)
        node_emb_1 = self.encoder(x_1, edge_index_1, edge_weights_1)
        node_emb_2 = self.encoder(x_2, edge_index_2, edge_weights_2)

        graph_emb = self.graph_embedding(node_emb, batch)
        graph_emb_1 = self.graph_embedding(node_emb_1, batch)
        graph_emb_2 = self.graph_embedding(node_emb_2, batch)

        return graph_emb, graph_emb_1, graph_emb_2


class GCL_NoDA(GCNEncoderDecoderClassifier):
    def forward(self, data):
        # sample minibatch of examples x1,...xN
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index_sc,
            data.edge_weight_sc,
            data.batch,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            x, edge_index, edge_weights, batch = (
                x.to(device),
                edge_index.to(device),
                edge_weights.to(device),
                batch.to(device),
            )
        else:
            device = torch.device("cpu")

        # embed example with encoder f to get z1^,...,zN^, z1',..., zN'
        node_emb = self.encoder(x, edge_index, edge_weights)
        node_emb_1 = self.encoder(x, edge_index, edge_weights)
        node_emb_2 = self.encoder(x, edge_index, edge_weights)

        graph_emb = self.graph_embedding(node_emb, batch)
        graph_emb_1 = self.graph_embedding(node_emb_1, batch)
        graph_emb_2 = self.graph_embedding(node_emb_2, batch)

        return graph_emb, graph_emb_1, graph_emb_2


class GCL_NoDA_Decoder(GCNEncoderDecoderClassifier):
    def forward(self, data):
        # sample minibatch of examples x1,...xN
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index_sc,
            data.edge_weight_sc,
            data.batch,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            x, edge_index, edge_weights, batch = (
                x.to(device),
                edge_index.to(device),
                edge_weights.to(device),
                batch.to(device),
            )
        else:
            device = torch.device("cpu")

        # embed example with encoder f to get z1^,...,zN^, z1',..., zN'
        node_emb = self.encoder(x, edge_index, edge_weights)
        node_emb_1 = self.encoder(x, edge_index, edge_weights)
        node_emb_2 = self.encoder(x, edge_index, edge_weights)

        graph_emb = self.graph_embedding(node_emb, batch)
        graph_emb_1 = self.graph_embedding(node_emb_1, batch)
        graph_emb_2 = self.graph_embedding(node_emb_2, batch)

        reconstructed_adj = self.decoder(node_emb)

        return reconstructed_adj, graph_emb, graph_emb_1, graph_emb_2


class GCL_FeatureMaskingEdgeDroppingDecoder(GCNEncoderDecoderClassifier):
    def forward(self, data):
        # sample minibatch of examples x1,...xN
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index_sc,
            data.edge_weight_sc,
            data.batch,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            x, edge_index, edge_weights, batch = (
                x.to(device),
                edge_index.to(device),
                edge_weights.to(device),
                batch.to(device),
            )
        else:
            device = torch.device("cpu")

        # augment each example twice to get x1^, ..., xN^, x1',...,xN'
        x_1, edge_index_1, edge_weights_1 = self._remove_edges(
            x, edge_index, edge_weights, self.prob_of_diappearance
        )
        x_2, edge_index_2, edge_weights_2 = self._substract_random_ones(
            x, edge_index, edge_weights, self.max_substraction
        )

        # embed example with encoder f to get z1^,...,zN^, z1',..., zN'
        node_emb = self.encoder(x, edge_index, edge_weights)
        node_emb_1 = self.encoder(x_1, edge_index_1, edge_weights_1)
        node_emb_2 = self.encoder(x_2, edge_index_2, edge_weights_2)

        graph_emb = self.graph_embedding(node_emb, batch)
        graph_emb_1 = self.graph_embedding(node_emb_1, batch)
        graph_emb_2 = self.graph_embedding(node_emb_2, batch)

        reconstructed_adj = self.decoder(node_emb)

        return reconstructed_adj, graph_emb, graph_emb_1, graph_emb_2


def load_pretrained_model(
    model_path, hidden_dims, num_classes, data_aug, model_mapping
):

    model_class, _ = model_mapping.get(data_aug)
    model = model_class(hidden_dims=hidden_dims, num_classes=num_classes)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class FineTuneClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, normalize=False):
        super(FineTuneClassifier, self).__init__()

        # Defining the network structure based on hidden_dim
        if hidden_dim == 1:
            self.fc1 = torch.nn.Linear(input_dim, output_dim)
            self.fc2 = None  # No second layer needed
        elif hidden_dim == 16:
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # Normalization parameter
        self.normalize = normalize

    def forward(self, x):
        # Normalize embeddings if the parameter 'normalize' is True
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        x = self.fc1(x)

        # Apply the second layer if it exists
        if self.fc2 is not None:
            x = self.fc2(x)
        return x
