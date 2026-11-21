import click as ck
import torch as th
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import dgl
from multiprocessing import Pool
from functools import partial

from tool.utils import Ontology, propagate_annots
from tool.model import RGCNWithGOGeometry, MLPModel
from tool.data import load_ppi_data, load_normal_forms
from tool.torch_utils import FastTensorDataLoader


class SoftWeightedFusion(nn.Module):
    def __init__(self, n_views=2):
        super().__init__()
        self.weights = nn.Parameter(th.ones(n_views))

    def forward(self, preds_list):
        weights = F.softmax(self.weights, dim=0)  # 归一化为和为1
        weighted_sum = sum(weights[i] * preds_list[i] for i in range(len(preds_list)))
        return weighted_sum, weights


@ck.command()
@ck.option('--data-root', '-dr', default='data', help='Data folder')
@ck.option('--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']), help='GO subontology')
@ck.option('--model-name-geo', '-mg', default='gingo_geo', help='Geometric model name (without .th)')
@ck.option('--model-name-mlp', '-mm', default='mlp_esm', help='MLP model name (without .th)')
@ck.option('--fusion-model-name', '-fm', default='fusion_geo_mlp', help='Fusion model name for saving')
@ck.option('--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']), help='Test set (only test or nextprot)')
@ck.option('--batch-size', '-bs', default=64, help='Batch size for inference')
@ck.option('--epochs', '-ep', default=20, help='Training epochs for fusion weights')
@ck.option('--device', '-d', default='cuda:0', help='Device')
def main(data_root, ont, model_name_geo, model_name_mlp, fusion_model_name,
         test_data_name, batch_size, epochs, device):

    # -------------------- 1. 准备必要配置 --------------------
    go_file = f'{data_root}/go.obo'
    go_norm_file = f'{data_root}/go.norm'  # 或 go-plus.norm
    go = Ontology(go_file, with_rels=True)

    features_length = 2560
    features_column = 'esm2'
    test_data_file = f'{test_data_name}_data.pkl'
    ppi_graph_file = f'ppi_{test_data_name}.bin'   # 需要包含训练+验证+测试全部节点

    # 加载图数据（含特征和标签，以及节点划分）
    mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
        data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)

    n_terms = len(terms_dict)
    # 加载GO范式以初始化几何模型
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(go_norm_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    # 移到设备
    device = th.device(device)
    labels = labels.to(device)
    graph = graph.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    # -------------------- 2. 加载两个训练好的模型 --------------------
    # 几何模型
    n_edge_types = int(graph.edata['etypes'].max().item()) + 1 if graph.num_edges() > 0 else 1

    geo_model = RGCNWithGOGeometry(
        input_length=features_length,
        nb_gos=n_terms,
        nb_zero_gos=n_zeros,
        nb_rels=n_rels,
        n_edge_types=n_edge_types,
        device=device,
        rgcn_hidden_dim=1024,
        go_embed_dim=1024,
        margin=0.1,
        dropout=0.3,
    ).to(device)
    geo_model_file = f'{data_root}/{ont}/{model_name_geo}.th'
    geo_model.load_state_dict(th.load(geo_model_file, map_location=device))
    geo_model.eval()

    # MLP模型
    mlp_model = MLPModel(features_length, n_terms, device).to(device)
    mlp_model_file = f'{data_root}/{ont}/{model_name_mlp}.th'
    mlp_model.load_state_dict(th.load(mlp_model_file, map_location=device))
    mlp_model.eval()

    print("Generating predictions on validation set...")

    # 几何模型验证集推理
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2, prefetch_node_feats=['feat'])
    valid_loader_geo = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler, batch_size=batch_size, shuffle=False, drop_last=False)

    geo_valid_preds = []
    with th.no_grad():
        for _, output_nodes, blocks in valid_loader_geo:
            logits = geo_model(blocks)
            geo_valid_preds.append(logits.cpu())
    geo_valid_preds = th.cat(geo_valid_preds, dim=0).numpy()

    # MLP模型验证集推理
    valid_features = data[valid_nids.cpu()].to(device)
    valid_loader_mlp = FastTensorDataLoader(valid_features, batch_size=batch_size, shuffle=False)
    mlp_valid_preds = []
    with th.no_grad():
        for batch_features, in valid_loader_mlp:
            logits = mlp_model(batch_features)
            mlp_valid_preds.append(logits.cpu())
    mlp_valid_preds = th.cat(mlp_valid_preds, dim=0).numpy()

    valid_labels = labels[valid_nids].cpu().numpy()  # 真实的验证集标签

    # -------------------- 4. 直接读取训练时保存的测试集预测文件 --------------------
    print("Loading precomputed test predictions...")

    # 几何模型的测试预测
    geo_test_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name_geo}.pkl'
    geo_test_df = pd.read_pickle(geo_test_file)
    geo_test_preds = np.array(geo_test_df['preds'].tolist())

    # MLP模型的测试预测
    mlp_test_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name_mlp}.pkl'
    mlp_test_df = pd.read_pickle(mlp_test_file)
    mlp_test_preds = np.array(mlp_test_df['preds'].tolist())

    test_labels = labels[test_nids].cpu().numpy()

    valid_preds_list = [th.tensor(geo_valid_preds, dtype=th.float32),
                        th.tensor(mlp_valid_preds, dtype=th.float32)]
    valid_labels_t = th.tensor(valid_labels, dtype=th.float32)

    fusion = SoftWeightedFusion(n_views=2).to(device)
    optimizer = Adam(fusion.parameters(), lr=1e-2)
    criterion = nn.BCELoss()

    print("Training fusion weights on validation set...")
    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        fusion.train()
        # 将数据移到GPU
        preds_cuda = [p.to(device) for p in valid_preds_list]
        labels_cuda = valid_labels_t.to(device)

        # 前向：融合
        fused, weights = fusion(preds_cuda)
        loss = criterion(fused, labels_cuda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 评估当前权重在验证集上的loss
        fusion.eval()
        with th.no_grad():
            fused_val, _ = fusion(preds_cuda)  # 用同一批数据，但这里全量已验证集不大
            val_loss = criterion(fused_val, labels_cuda).item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}, Weights: {weights.detach().cpu().numpy()}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = weights.detach().cpu().numpy()

    print(f"Best weights: {best_weights}")

    test_preds_list = [th.tensor(geo_test_preds, dtype=th.float32),
                       th.tensor(mlp_test_preds, dtype=th.float32)]
    fused_test = best_weights[0] * test_preds_list[0] + best_weights[1] * test_preds_list[1]
    fused_test = fused_test.numpy()

    print("Propagating annotations...")
    preds_list = list(fused_test)
    with Pool(32) as p:
        preds_list = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds_list)

    # 用 geo_test_df 作为模板（包含所有列），替换 preds 列
    geo_test_df['preds'] = preds_list
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{fusion_model_name}.pkl'
    geo_test_df.to_pickle(out_file)
    print(f"Saved fused predictions to {out_file}")

if __name__ == '__main__':
    main()