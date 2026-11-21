import click as ck
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
import math
import dgl
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from tool.utils import Ontology, propagate_annots
from tool.model import RGCNWithGOGeometry
from tool.data import load_ppi_data, load_normal_forms
from tool.metrics import compute_roc


@ck.command()
@ck.option('--data-root', '-dr', default='data', help='Data folder')
@ck.option('--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']), help='GO subontology')
@ck.option('--model-name', '-m', default='rgcn_geo', help='Prediction model name')
@ck.option('--model-id', '-mi', type=int, required=False, help='Model ID for versioning')
@ck.option('--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot', 'valid']),
           help='Test data set name')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for training')
@ck.option('--epochs', '-ep', default=256, help='Training epochs')
@ck.option('--load', '-ld', is_flag=True, help='Load Model?')
@ck.option('--device', '-d', default='cuda:0', help='Device')
def main(data_root, ont, model_name, model_id, test_data_name, batch_size, epochs, load, device):

    if model_id is not None:
        model_name = f'{model_name}_{model_id}'


    if model_name.find('plus') != -1:
        go_norm_file = f'{data_root}/go-plus.norm'
    else:
        go_norm_file = f'{data_root}/go.norm'

    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{model_name}.th'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'
    go = Ontology(go_file, with_rels=True)


    features_length = 2560
    features_column = 'esm2'
    ppi_graph_file = f'ppi_{test_data_name}.bin'
    test_data_file = f'{test_data_name}_data.pkl'

    mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
        data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)

    n_terms = len(terms_dict)
    valid_labels = labels[valid_nids].cpu().numpy()
    test_labels = labels[test_nids].cpu().numpy()
    labels = labels.to(device)
    graph = graph.to(device)
    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_norm_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = (nf1, nf2, nf3, nf4)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2, prefetch_node_feats=['feat'])
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0)
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    n_edge_types = int(graph.edata['etypes'].max().item()) + 1 if graph.num_edges() > 0 else 1
    print("etypes unique:", graph.edata['etypes'].unique().cpu().tolist())
    print("etypes bincount:", graph.edata['etypes'].bincount().cpu().tolist())
    print("n_edge_types:", n_edge_types)

    net = RGCNWithGOGeometry(
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
    print(net)

    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            with tqdm(total=train_steps) as bar:
                for input_nodes, output_nodes, blocks in train_dataloader:
                    bar.update(1)
                    logits = net(blocks)
                    batch_labels = labels[output_nodes]
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    el_loss = net.el_loss(normal_forms)
                    total_loss = loss + el_loss
                    train_loss += loss.detach().item()
                    train_elloss = el_loss.detach().item()

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                with tqdm(total=valid_steps) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(blocks)
                        batch_labels = labels[output_nodes]
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds.append(logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                preds = np.concatenate(preds)
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, EL Loss - {train_elloss}, '
                      f'Valid loss - {valid_loss}, AUC - {roc_auc}')

            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()

    # ---------- 5. 加载最佳模型并评估 ----------
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()

    with th.no_grad():
        valid_steps = int(math.ceil(len(valid_nids) / batch_size))
        valid_loss = 0
        preds = []
        with tqdm(total=valid_steps) as bar:
            for input_nodes, output_nodes, blocks in valid_dataloader:
                bar.update(1)
                logits = net(blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                valid_loss += batch_loss.detach().item()
                preds.append(logits.detach().cpu().numpy())
        valid_loss /= valid_steps
        preds_valid = np.concatenate(preds)

    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        with tqdm(total=test_steps) as bar:
            for input_nodes, output_nodes, blocks in test_dataloader:
                bar.update(1)
                logits = net(blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
        test_loss /= test_steps
        preds_test = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds_test)

    print(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}')
    with open(f'{data_root}/{ont}/valid_{model_name}.pf', 'w') as f:
        f.write(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}\n')

    # 传播注释（利用 GO 结构）
    preds_list = list(preds_test)
    with Pool(32) as p:
        preds_list = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds_list)
    test_df['preds'] = preds_list
    test_df.to_pickle(out_file)


if __name__ == '__main__':
    main()
