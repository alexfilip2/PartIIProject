from gat_impl.KerasAttentionHead import *
import numpy as np
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from utils.ToolsDataProcessing import adj_to_bias


class TensorflowGraphGAT(object):
    @staticmethod
    def average_feature_aggregator(model_gat_output, **kwargs):
        '''
          Averages the node features produced by GAT and feeds them into a single layer of MLP.
        :param model_gat_output: Keras layer with output shape (?, nb_nodes, F') storing the GAT output node features
        :param kwargs: additional parameters used by the MLP: decay rate, targeted dimensionality
        :return: The final output of the architecture, Keras layer of output shape (?, F') where F' is the number
        of traits predicted at once.
        '''

        def reduce_sum(layer):
            return tf.reduce_sum(layer, axis=1)

        in_mlp = Lambda(function=reduce_sum)(model_gat_output)
        out_avg = Dense(units=kwargs['target_score_type'],
                        kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                        use_bias=True)(in_mlp)
        return out_avg

    @staticmethod
    def concat_feature_aggregator(model_gat_output, **kwargs):
        '''
         Concatenates the node features produced by GAT and feeds them into a single layer of MLP.
        :param model_gat_output: Keras layer with output shape (?, nb_nodes, F') storing the GAT output node features
        :param kwargs: additional parameters used by the MLP: decay rate, targeted dimensionality
        :return: The final output of the architecture, Keras layer of output shape (?, F') where F' is the number
        of traits predicted at once.
        '''

        # have to use dynamic tensor shape due to variable batch sizes
        def concat(layer):
            dyn_shape = layer.get_shape().as_list()
            concat_axis = np.prod(dyn_shape[1:])
            return tf.reshape(layer, [-1, concat_axis])

        in_mlp = Lambda(function=concat)(model_gat_output)
        # fed the result into MLP layer with separate weights for each personality score
        out_concat = Dense(units=kwargs['target_score_type'],
                           kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                           use_bias=True)(in_mlp)
        return out_concat

    @staticmethod
    def master_node_aggregator(model_gat_output, **kwargs):
        '''
         Aggregate the node features produced by GAT via master-node and feeds them into a single layer of MLP.
        :param model_gat_output: Keras layer with output shape (?, nb_nodes, F') storing the GAT output node features
        :param kwargs: additional parameters used by the MLP: decay rate, targeted dimensionality, graph order
        :return: The final output of the architecture, Keras layer of output shape (?, F') where F' is the number
        of traits predicted at once.
        '''

        def attach_master(nb_nodes):
            mast_mat = np.zeros((nb_nodes, nb_nodes))
            for i in range(nb_nodes):
                mast_mat[nb_nodes - 1][i] = 1.0
                mast_mat[i][nb_nodes - 1] = 1.0
            return np.expand_dims(mast_mat, axis=0), np.expand_dims(adj_to_bias(mast_mat), axis=0)

        # connect the master to all the other nodes and pairwise disconnect them
        extended_adj, extended_bias = attach_master(kwargs['dim_nodes'])

        def prepare_adjacency(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_adj, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        def prepare_biases(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_bias, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        batch_ext_adjs = Lambda(function=prepare_adjacency)(model_gat_output)
        batch_ext_bias = Lambda(function=prepare_biases)(model_gat_output)

        # master GAT layer
        master_layer = GATLayer(F_=kwargs['master_feats'],
                                attn_heads=kwargs['master_heads'],
                                attn_heads_reduction='concat',
                                flag_batch_norm=True,
                                flag_edge_weights=False,
                                dropout_rate=kwargs['attn_drop'],
                                decay_rate=kwargs['decay_rate'],
                                activation=kwargs['non_linearity'])([model_gat_output, batch_ext_adjs, batch_ext_bias])

        # extract only the high-level features produced for the master node (in each batch graph)
        def extract_master_feats(inputs):
            return tf.squeeze(tf.slice(input_=inputs, begin=[0, kwargs['dim_nodes'] - 1, 0],
                                       size=[tf.shape(inputs)[0], 1, kwargs['master_feats']]), axis=1)

        in_mlp = Lambda(function=extract_master_feats)(master_layer)
        # attach the last layer as a MLP
        out_master = Dense(units=kwargs['target_score_type'],
                           kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                           use_bias=True)(in_mlp)

        return out_master

    @staticmethod
    def inference_keras(dim_nodes, dim_feats, hidden_units, attention_heads, attn_drop, decay_rate,
                        use_batch_norm, include_ew, non_linearity, readout_aggregator, target_score_type, **kwargs):
        '''
         Builds the GAT Keras architecture.
        :param dim_nodes: graph order of the graphs received as input in the batch
        :param dim_feats: initial node features dimension
        :param hidden_units:  dimensions of the node feature produced by each attention head per each layer
        :param attention_heads: number of attention heads on each layer
        :param attn_drop: dropout rate applied on the input features (of previous layer) and the alpha_ij coefficients
        :param decay_rate: rate of decay for the L2 and robustness regularizations
        :param use_batch_norm: flag for using batch normalization on each layer
        :param include_ew: flag for integrating the edge weights
        :param non_linearity: activation function of the layer applied to the feature vectors element-wise
        :param readout_aggregator: aggregation function for all the node features produced by the underlying GAT
        :param target_score_type:  dimension of the GAT NN target, the number of traits predicted at once
        :param kwargs: used to accept any dictionary of actual parameters
        :return: Keras GAT model built using the Functional API
        '''
        # Define the input placeholders into which data will be fed (batch shape included by default)
        batch_node_features = Input(shape=(dim_nodes, dim_feats), name='node_features')
        batch_adj_mats = Input(shape=(dim_nodes, dim_nodes), name='adjacency_matrices')
        batch_bias_mats = Input(shape=(dim_nodes, dim_nodes), name='masks')

        # Integrate the hyper-parameters of the arch. depending on if master-node or not
        mutable_hu = hidden_units.copy()
        mutable_ah = attention_heads.copy()
        layer_args = {'dropout_rate': attn_drop,
                      'flag_batch_norm': use_batch_norm,
                      'decay_rate': decay_rate,
                      'flag_edge_weights': include_ew,
                      'attn_heads_reduction': 'concat',
                      'activation': non_linearity}
        master_heads = master_feats = 0
        if readout_aggregator is TensorflowGraphGAT.master_node_aggregator:
            master_heads = mutable_ah.pop()
            master_feats = mutable_hu.pop()

        # Input GAT layer
        layer_args.update({'F_': mutable_hu[0], 'attn_heads': mutable_ah[0]})
        input_layer = GATLayer(**layer_args)([batch_node_features, batch_adj_mats, batch_bias_mats])

        # Hidden GAT layers
        prev_out = input_layer
        for i in range(1, len(mutable_ah) - 1):
            layer_args.update({'F_': mutable_hu[i], 'attn_heads': mutable_ah[i]})
            i_th_layer = GATLayer(**layer_args)(inputs=[prev_out, batch_adj_mats, batch_bias_mats])
            prev_out = i_th_layer
            # accumulate the regularization losses

        # Output GAT layer
        layer_args.update({'F_': mutable_hu[-1], 'attn_heads': mutable_ah[-1]})
        last_layer = GATLayer(**layer_args)(inputs=[prev_out, batch_adj_mats, batch_bias_mats])

        # aggregate all the output node features using the specified strategy
        gat_output = readout_aggregator(model_gat_output=last_layer,
                                        master_heads=master_heads,
                                        master_feats=master_feats,
                                        target_score_type=target_score_type,
                                        dim_nodes=dim_nodes,
                                        attn_drop=attn_drop,
                                        decay_rate=decay_rate,
                                        non_linearity=non_linearity)
        # define the Keras Model based on the Functional API
        model = Model(outputs=[gat_output], inputs=[batch_node_features, batch_adj_mats, batch_bias_mats])
        model.summary()
        return model
