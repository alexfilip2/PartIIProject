import networkx as nx
from node2vec import Node2Vec
from ToolsFunctional import *

node2vec_emb_dir = join(os.getcwd(), os.pardir, 'PartIIProject', 'node2vec_embeds')
if not os.path.exists(node2vec_emb_dir):
    os.makedirs(node2vec_emb_dir)


def create_node_embedding(matrices_dim=50, session_id=1):
    node2vec_emb_dir = join(os.getcwd(), os.pardir, 'PartIIProject', 'node2vec_embeds',
                            'emb_dim%d_sess%d' % (matrices_dim, session_id))
    dict_funct_adjs = get_functional_adjs(matrices_dim)
    # Create a graph
    for subj_id in dict_funct_adjs.keys():
        graph = nx.from_numpy_matrix(A=get_binary_adj(dict_funct_adjs[subj_id]))
        # FILES
        EMBEDDING_FILENAME = join(node2vec_emb_dir, subj_id + 'embeddings.emb')
        if os.path.exists(EMBEDDING_FILENAME): continue
        EMBEDDING_MODEL_FILENAME = './embeddings.model'

        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(graph, dimensions=8, walk_length=matrices_dim, num_walks=100, workers=4)
        # Use temp_folder for big graphs

        # Embed nodes
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

        # Save embeddings for later use
        model.wv.save_word2vec_format(EMBEDDING_FILENAME)

        # Save model for later use
        model.save(EMBEDDING_MODEL_FILENAME)

