import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip

from ogb.io.read_graph_raw import read_csv_graph_raw, read_binary_graph_raw
from ogb.graphproppred import DglGraphPropPredDataset
from tqdm import tqdm
from utils.basis_transform import basis_transform
import time
from utils.config import get_config_from_json


class DglGraphPropPred(DglGraphPropPredDataset):
    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if self.task_type == 'subtoken prediction':
            target_sequence_file_path = osp.join(processed_dir, 'target_sequence')

        if os.path.exists(pre_processed_file_path):
            print('Loading the cached file... (NOTE: delete it if you change the preprocessing settings)')

            if self.task_type == 'subtoken prediction':
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']

        else:
            print('Converting graphs into DGL objects...')

            ### check download
            if self.binary:
                # npz format
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'data.npz'))
            else:
                # csv file
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz'))

            ### download
            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root, self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            ### preprocess
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')

            graphs = read_graph_dgl(raw_dir, add_inverse_edge=add_inverse_edge,
                                    additional_node_files=additional_node_files,
                                    additional_edge_files=additional_edge_files, binary=self.binary)

            if self.task_type == 'subtoken prediction':
                # the downloaded labels are initially joined by ' '
                labels_joined = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                            header=None).values
                # need to split each element into subtokens
                labels = [str(labels_joined[i][0]).split(' ') for i in range(len(labels_joined))]

                print('Saving...')
                save_graphs(pre_processed_file_path, graphs)
                torch.save(labels, target_sequence_file_path)

                ### load preprocessed files
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                if self.binary:
                    labels = np.load(osp.join(raw_dir, 'graph-label.npz'))['graph_label']
                else:
                    labels = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                         header=None).values

                has_nan = np.isnan(labels).any()

                if 'classification' in self.task_type:
                    if has_nan:
                        labels = torch.from_numpy(labels).to(torch.float32)
                    else:
                        labels = torch.from_numpy(labels).to(torch.long)
                else:
                    labels = torch.from_numpy(labels).to(torch.float32)

                print('Saving...')
                save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

                ### load preprocessed files
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']


def read_graph_dgl(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[], binary=False):
    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files)

    dgl_graph_list = []

    print('Converting graphs into DGL objects...')

    config = get_config_from_json("./ogbg-molpcba.json")
    basis = config.basis
    epsilon = config.epsilon
    power = config.power
    print('Basis configurations: basis: {}, epsilon: {}, power: {}'.format(basis, epsilon, power))

    # trans_start = time.time()
    for graph in tqdm(graph_list):
        g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])

        if graph['edge_feat'] is not None:
            g.edata['feat'] = torch.from_numpy(graph['edge_feat'])

        if graph['node_feat'] is not None:
            g.ndata['feat'] = torch.from_numpy(graph['node_feat'])

        for key in additional_node_files:
            g.ndata[key[5:]] = torch.from_numpy(graph[key])

        for key in additional_edge_files:
            g.edata[key[5:]] = torch.from_numpy(graph[key])

        g = dgl.remove_self_loop(g)
        # _num_edges = g.edata['feat'].shape[0]
        g.edata['feat'] = g.edata['feat'] + 1  # Caution! padding with zeros in edge attr.
        g = dgl.add_self_loop(g)
        # assert (g.edata['feat'].shape[0] == _num_edges + g.num_nodes())
        g = basis_transform(g, basis=basis, epsilon=epsilon, power=power)
        # g = dgl.remove_self_loop(g)

        dgl_graph_list.append(g)
    # total_time = time.time() - trans_start
    # print('Basis transformation total and avg time:', total_time, total_time / len(graph_list))

    return dgl_graph_list
