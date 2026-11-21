import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from .base import BaseDeepGOModel, MLPBlock, Residual


class RGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_relations, dropout=0.3, num_bases=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        bases = num_bases or min(n_relations, 4)

        self.conv1 = RelGraphConv(
            hidden_dim, hidden_dim, n_relations,
            regularizer="basis",
            num_bases=bases,
            activation=F.gelu,
            self_loop=True,
            dropout=dropout,
            layer_norm=True,
        )
        self.conv2 = RelGraphConv(
            hidden_dim, hidden_dim, n_relations,
            regularizer="basis",
            num_bases=bases,
            activation=F.gelu,
            self_loop=True,
            dropout=dropout,
            layer_norm=True,
        )
        self.projection = None

    @staticmethod
    def _source_features(block):
        if "feat" in block.srcdata:
            return block.srcdata["feat"]
        return block.ndata["feat"]["_N"]

    @staticmethod
    def _edge_types(block):
        if "etypes" not in block.edata:
            raise KeyError("block.edata['etypes'] 不存在，检查图或采样器")
        return block.edata["etypes"].to(dtype=th.long)

    def forward(self, blocks):
        x = self.input_projection(self._source_features(blocks[0]))
        x = self.conv1(blocks[0], x, self._edge_types(blocks[0]))
        x = self.conv2(blocks[1], x, self._edge_types(blocks[1]))
        if self.projection is not None:
            x = self.projection(x)
        return x

    def set_projection(self, target_dim):
        if self.hidden_dim != target_dim:
            self.projection = nn.Linear(self.hidden_dim, target_dim)
        else:
            self.projection = nn.Identity()


class RGCNWithGOGeometry(BaseDeepGOModel):
    def __init__(
        self,
        input_length,
        nb_gos,
        nb_zero_gos,
        nb_rels,
        n_edge_types,
        device,
        rgcn_hidden_dim=1024,
        go_embed_dim=1024,
        margin=0.1,
        dropout=0.3,
    ):
        super().__init__(
            input_length, nb_gos, nb_zero_gos, nb_rels, device,
            hidden_dim=go_embed_dim, embed_dim=go_embed_dim, margin=margin,
        )
        self.rgcn_extractor = RGCNEncoder(
            input_dim=input_length,
            hidden_dim=rgcn_hidden_dim,
            n_relations=n_edge_types,
            dropout=dropout,
        )
        self.rgcn_extractor.set_projection(go_embed_dim)

    def forward(self, blocks):
        protein_embed = self.rgcn_extractor(blocks)   # [batch, go_embed_dim]
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        logits = th.matmul(protein_embed, hasFuncGO.T) + go_rad
        return th.sigmoid(logits)


class MLPModel(nn.Module):
    """
    Baseline MLP model with two fully connected layers with residual connection
    """

    def __init__(self, input_length, nb_gos, device, nodes=[1024, ]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, features):
        return self.net(features)

