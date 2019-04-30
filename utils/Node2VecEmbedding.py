import networkx as nx
from node2vec import Node2Vec
from utils.LoadFunctionalData import get_functional_adjacency
from utils.ToolsDataProcessing import get_binary_adj
import os

root_node2vec_dir = os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'Data', 'node2vec_embeds')
if not os.path.exists(root_node2vec_dir):
    os.makedirs(root_node2vec_dir)


def create_node_embedding(matrices_dim: int = 50, session_id: int = 1):
    '''
     Creates the Node2Vec embeddings (node features) for all the functional adjacency matrices
    :param matrices_dim: int specifying the dimension of the loaded feature matrices
    :param session_id: int the scan session from which they were generated
    :return: void, it writes the embedding files on disk
    '''
    node2vec_emb_dir = os.path.join(root_node2vec_dir, 'emb_dim%d_sess%d' % (matrices_dim, session_id))
    dict_adj = get_functional_adjacency(matrices_dim, session_id)
    # Create a networkx graph
    for subject in dict_adj.keys():
        graph = nx.from_numpy_matrix(A=get_binary_adj(dict_adj[subject]))
        # specify a file for the target of the embedding
        embedding_path = os.path.join(node2vec_emb_dir, subject + 'embeddings.emb')
        if os.path.exists(embedding_path):
            continue
        node2vec_model = './embeddings.model'
        # define the Node2Vec model
        node2vec = Node2Vec(graph, dimensions=8, walk_length=matrices_dim, num_walks=100, workers=4)

        # Unsupervised fitting on the input graph
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Save the embedding generated for the input adjacency matrix
        model.wv.save_word2vec_format(embedding_path)

        # Save model for later use
        model.save(node2vec_model)
