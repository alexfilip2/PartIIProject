import numpy as np

np.set_printoptions(threshold=np.nan)
import os
import pandas as pd
import random

struct_mat_root = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'structural_mat')
struct_mat_HCP = os.path.join(struct_mat_root, 'PTN matrices HCP')
struct_feat = os.path.join(struct_mat_root, 'Features_all.xlsx')

pers_scores = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'TIVscores',
                           '1016_HCP_withTIV_acorrected_USETHIS.xlsx')


def get_struct_nodes():
    df = pd.read_excel(struct_feat, sheet_name='nodes', index_col=None, header=None)

    brain_reg_names = list(df[1])
    brain_reg_ids = list(df[0])
    nodes_dict = dict(zip(brain_reg_names, brain_reg_ids))
    return nodes_dict


def get_feat_names():
    df = pd.read_excel(struct_feat, sheet_name='features', index_col=None, header=None)
    features_names = sorted(list(df[0]))
    return features_names


def get_graphs_feat():
    df = pd.read_excel(struct_feat, sheet_name='Data')
    graph_feats = {}
    feats = get_feat_names()
    nodes = get_struct_nodes()
    for index, row in df.iterrows():
        node_feat_vecs = {}
        for node in nodes.keys():
            curr_node_feat = []
            for feat_name in feats:
                    if ('fs'+ node + '_' + feat_name) in row.keys():
                        curr_node_feat.append(float(row['fs'+ node + '_' + feat_name]))
            node_feat_vecs[node] = curr_node_feat
        graph_feats[str(int(row['Subjects']))] = node_feat_vecs

    return graph_feats




def get_filter_strcut_adjs():
    all_sub_dirs = [os.path.join(struct_mat_HCP,subdir) for subdir in next(os.walk(struct_mat_HCP))[1]]
    filtered_nodes =  get_struct_nodes().values()

    adj = {}
    for subdir in all_sub_dirs:
        for subj_id in os.listdir(subdir):
            with open(os.path.join(subdir,subj_id), 'r', encoding='UTF-8') as data:
                graph = []
                for row,line in enumerate(data):
                    if row not in filtered_nodes: continue
                    adj_row = []
                    for column, edge_weight in enumerate(line.split()):
                        if column not in filtered_nodes:continue
                        adj_row.append(float(edge_weight))
                    graph.append(adj_row)


            adj[subj_id] = np.array(graph)

    return  adj

if __name__ == "__main__":
    #feats = get_graphs_feat()
    print(len(get_filter_strcut_adjs().keys()))
