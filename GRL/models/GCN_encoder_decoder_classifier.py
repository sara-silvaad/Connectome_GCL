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
        concatenate=True,
        model_name="BaseModel",
        num_nodes=87,
        negs=False,
        permutar=False,
        take=False,
    ):
        super(GCNEncoderDecoderClassifier, self).__init__()
        self.model_name = model_name
        self.num_nodes = num_nodes
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.concatenate = concatenate
        self.convs = torch.nn.ModuleList()
        self.negs = negs
        self.permute = permutar
        self.take = take

        for i in range(len(hidden_dims) - 1):
            conv = GCNConv(hidden_dims[i], hidden_dims[i + 1])
            self.convs.append(conv)

        if self.pooling_type == "ave":
            self.classifier = torch.nn.Linear(
                hidden_dims[-1] if not concatenate else sum(hidden_dims[1:]),
                num_classes,
            )
        elif self.pooling_type == "ave_on_nodes":
            self.classifier = torch.nn.Linear(self.num_nodes, num_classes)
        elif self.pooling_type == "concatenate":
            self.classifier = torch.nn.Linear(
                87 * hidden_dims[-1] if not concatenate else 87 * sum(hidden_dims[1:]),
                num_classes,
            )

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


class EncoderDecoderSCFC(GCNEncoderDecoderClassifier):
    def forward(self, data):
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

        node_emb = self.encoder(x, edge_index, edge_weights)
        reconstructed_adj = self.decoder(node_emb)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb).float())

        return reconstructed_adj, node_emb, classif_logits, graph_emb, x


class EncoderClassifierSC(GCNEncoderDecoderClassifier):
    def forward(self, data):
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

        node_emb = self.encoder(x, edge_index, edge_weights)
        graph_emb = self.graph_embedding(node_emb, batch)
        classif_logits = torch.squeeze(self.classifier(graph_emb)).float()
        return classif_logits, x, graph_emb
