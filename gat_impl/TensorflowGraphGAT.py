from gat_impl.KerasAttentionHead import *
from utils.ToolsDataProcessing import attach_master
import numpy as np

# number of attention heads of the master node layer
MASTER_HEADS = 3


class TensorflowGraphGAT(object):

    def average_feature_aggregator(self, model_GAT_output, **kwargs):
        """ Averages the node features produced by GAT
                Parameters
                ----------
                model_GAT_output : tensor of shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each node of each graph in the batch

                Returns
                -------
                out_avg : tensor of shape (?, F'), F' is 1 (targets one personality trait)
                    The tensor storing the average of the feat. vectors across the entire node set
        """
        out_avg = tf.reduce_sum(model_GAT_output, axis=1)
        # output = tf.layers.dense(output, units=target_score_type, use_bias=False)
        return out_avg

    def concat_feature_aggregator(self, model_GAT_output, **kwargs):
        """ Concatenates the node features produced by GAT and feed them into a single layer of MLP
                Parameters
                ----------
                model_GAT_output : tensor of shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each node of each graph in the batch

                Returns
                -------
                out_mlp : tensor of shape (?, kwargs['target_score_type'])
                    The tensor storing the regression output for each graph in the batch
        """
        # run-time tensor shape as variable batch sizes are possible
        dyn_shape = model_GAT_output.get_shape().as_list()
        concat_axis = np.prod(dyn_shape[1:])
        out_concat = tf.reshape(model_GAT_output, [-1, concat_axis])
        # fed the result into MLP layer with separate weights for each personality score
        out_mlp = tf.layers.dense(out_concat, units=kwargs['target_score_type'], use_bias=False)
        return out_mlp

    def master_node_aggregator(self, model_GAT_output, **kwargs):
        """ Aggregate the node features produced by GAT via a Master node
                Parameters
                ----------
                model_GAT_output : tensor of shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each node of each graph in the batch

                Returns
                -------
                out_avg : tensor of shape (?, F'),
                            The tensor storing the average of the feat. vectors across the entire node set
        """
        init_nb_nodes = int(model_GAT_output.shape[1])
        dyn_batch_size = tf.shape(model_GAT_output)[0]
        # consider 0's feature vec for the master node
        master_feats = tf.zeros([1, model_GAT_output.shape[-1]])
        # create masters for each graph in the batch
        master_feats_tilled = tf.expand_dims(tf.tile(master_feats, tf.stack([dyn_batch_size, 1])), axis=1)
        # extended node feats including the masters (?,N+1,F')
        extended_feats = tf.concat([model_GAT_output, master_feats_tilled], axis=1)
        # connect the master to all the nodes and  pairwise disconnect these
        extended_adj, extended_bias = attach_master(init_nb_nodes)
        # repeat the same adjacency matrix for all the graphs in the batch
        extended_adj = tf.cast(tf.tile(extended_adj, [dyn_batch_size, 1, 1]), dtype=tf.float32)
        extended_bias = tf.cast(tf.tile(extended_bias, [dyn_batch_size, 1, 1]), dtype=tf.float32)

        # master GAT layer
        master_layer, _, _ = GraphAttention(F_=kwargs['target_score_type'],
                                            attn_heads=kwargs['master_heads'],
                                            attn_heads_reduction='average',
                                            dropout_rate=kwargs['attn_drop'],
                                            activation=lambda x: x)(
            inputs=[extended_feats, extended_adj, extended_bias, kwargs['is_train']], include_ew=False)

        # extract only the high-level features produced for the master node (in each batch graph)
        out_master = tf.squeeze(tf.slice(input_=master_layer, begin=[0, init_nb_nodes, 0],
                                         size=[dyn_batch_size, 1, kwargs['target_score_type']]), axis=1)

        return out_master

    def inference_keras(self, batch_node_features, batch_adj_mats, batch_bias_mats, hidden_units, attention_heads,
                        attn_drop, is_train, include_ew, non_linearity, readout_aggregator, target_score_type,
                        **kwargs):
        """ Builds the GAT architecture
                Parameters
                ----------
                batch_node_features : tensor of shape (?, nb_nodes, F)
                    The tensor storing the input node features for each graph in the batch
                batch_adj_mats: tensor of shape (?, nb_nodes, nb_nodes)
                    The tensor storing the input weighted adjacency matrices
                batch_bias_mats: tensor of shape (?, nb_nodes, nb_nodes)
                    The tensor representing the mask for node connectivity
                hid_units: list of int
                    The length of the node feature vecs produced by each attn head of each layer
                nb_heads: list of int
                    Number of attention heads on each layer of the GAT NN
                attn_drop : float
                    Dropout rate for the input features (of previous layer) and the alpha_ij coefficients
                is_train: bool
                    Flag that enables or not the dropout layers
                include_weights : bool tensor
                    Flag for integrating the EDGE WEIGHTS when generating the alpha_ij coefficients
                activation : function
                    The activation function of the layer applied to the feature vectors element-wise
                aggregator : function
                    The aggregation function for all the node features for the final score prediction
                target_score_type: int
                    The output dimension of the GAT NN: predicting some of the Big-Five personality scores

                Returns
                -------
                out_avg : tensor of shape (?, F'),
                            The tensor storing the average of the feat. vectors across the entire node set
        """
        # change the dimension of the final features produced if averaging is employed
        out_layer_feat_dim = hidden_units[-1]
        if readout_aggregator is TensorflowGraphGAT.average_feature_aggregator:
            out_layer_feat_dim = target_score_type

        # input GAT layer
        out_input_layer, arch_uloss, arch_eloss = GraphAttention(F_=hidden_units[0],
                                                                 attn_heads=attention_heads[0],
                                                                 flag_batch_norm=True,
                                                                 attn_heads_reduction='concat',
                                                                 dropout_rate=attn_drop,
                                                                 activation=non_linearity)(
            inputs=[batch_node_features, batch_adj_mats, batch_bias_mats, is_train], include_ew=include_ew)
        out = out_input_layer
        # hidden GAT layers
        for i in range(1, len(attention_heads) - 1):
            out_i_th_layer, layer_uloss, layer_eloss = GraphAttention(F_=hidden_units[i],
                                                                      attn_heads=attention_heads[i],
                                                                      attn_heads_reduction='concat',
                                                                      flag_batch_norm=False,
                                                                      dropout_rate=attn_drop,
                                                                      activation=non_linearity)(
                inputs=[out, batch_adj_mats, batch_bias_mats, is_train], include_ew=include_ew)
            out = out_i_th_layer
            # accumulate the regularization losses
            arch_uloss, arch_eloss = tf.add(arch_uloss, layer_uloss), tf.add(arch_eloss, layer_eloss)

        # output GAT layer
        gat_output, output_uloss, output_eloss = GraphAttention(F_=out_layer_feat_dim,
                                                                attn_heads=attention_heads[-1],
                                                                flag_batch_norm=False,
                                                                attn_heads_reduction='average',
                                                                dropout_rate=attn_drop,
                                                                activation=lambda x: x)(
            inputs=[out, batch_adj_mats, batch_bias_mats, is_train], include_ew=include_ew)

        # average the regularization losses by the total nr of attention heads
        nb_attn_heads = np.sum(np.array(attention_heads))
        arch_uloss = tf.divide(tf.add(arch_uloss, output_uloss), nb_attn_heads)
        arch_eloss = tf.divide(tf.add(arch_eloss, output_eloss), nb_attn_heads)

        # aggregate all the output node features using the specified strategy
        pred_out = readout_aggregator(self, model_GAT_output=gat_output, target_score_type=target_score_type,
                                      is_train=is_train,
                                      attn_drop=attn_drop, master_heads=MASTER_HEADS)

        return pred_out, arch_uloss, arch_eloss
