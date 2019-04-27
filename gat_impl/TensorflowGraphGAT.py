from gat_impl.KerasAttentionHead import *
from utils.ToolsDataProcessing import attach_master
import numpy as np
from keras.layers import Input, Lambda, Dense
from keras.models import Model


class TensorflowGraphGAT(object):
    @staticmethod
    def average_feature_aggregator(model_gat_output, **kwargs):
        """ Averages the node features produced by GAT and feeds them into a single layer of MLP
                Parameters
                ----------
                model_gat_output : Keras layer with output shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each graph in the batch

                kwargs : additional parameters used by the MLP: decay rate, target dimensionality

                Returns
                -------
                out_avg : Keras layer of output shape (?, F'), F' is kwargs['target_score_type']
                    The final output of the architecture
        """

        def reduce_sum(layer):
            return tf.reduce_sum(layer, axis=1)

        in_mlp = Lambda(function=reduce_sum)(model_gat_output)
        out_avg = Dense(units=kwargs['target_score_type'],
                        kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                        use_bias=True)(in_mlp)
        return out_avg

    @staticmethod
    def concat_feature_aggregator(model_gat_output, **kwargs):
        """ Concatenates the node features produced by GAT and feed them into a single layer of MLP
                Parameters
                ----------
                model_gat_output : Keras layer with output shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each graph in the batch

                Returns
                -------
                out_mlp : tensor of shape (?, kwargs['target_score_type'])
                    The tensor storing the regression output for each graph in the batch
        """

        # run-time tensor shape as variable batch sizes are possible
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
        """ Aggregate the node features produced by GAT via a Master node
                Parameters
                ----------
                model_gat_output : tensor of shape (?, nb_nodes, F')
                    The tensor storing the GAT output node feature vectors for each node of each graph in the batch

                Returns
                -------
                out_avg : tensor of shape (?, F'),
                            The tensor storing the average of the feat. vectors across the entire node set
        """
        # connect the master to all the nodes and  pairwise disconnect these
        extended_adj, extended_bias = attach_master(kwargs['dim_nodes'])

        def prepare_adjs(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_adj, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        def prepare_biases(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_bias, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        batch_ext_adjs = Lambda(function=prepare_adjs)(model_gat_output)
        batch_ext_bias = Lambda(function=prepare_biases)(model_gat_output)

        # master GAT layer
        master_layer = GATLayer(F_=kwargs['master_feats'],
                                attn_heads=kwargs['master_heads'],
                                attn_heads_reduction='average',
                                flag_batch_norm=False,
                                flag_include_ew=False,
                                flag_edge_weights=False,
                                dropout_rate=kwargs['attn_drop'],
                                decay_rate=kwargs['decay_rate'],
                                activation=lambda x: x)([model_gat_output, batch_ext_adjs, batch_ext_bias])

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
        """ Builds the GAT architecture
                Parameters
                ----------
                dim_nodes : tensor of shape (?, nb_nodes, F)
                    The tensor storing the input node features for each graph in the batch
                dim_feats: tensor of shape (?, nb_nodes, nb_nodes)
                    The tensor storing the input weighted adjacency matrices
                hidden_units: list of int
                    The length of the node feature vecs produced by each attn head of each layer
                attention_heads: list of int
                    Number of attention heads on each layer of the GAT NN
                attn_drop : float
                    Dropout rate for the input features (of previous layer) and the alpha_ij coefficients
                include_ew : bool tensor
                    Flag for integrating the EDGE WEIGHTS when generating the alpha_ij coefficients
                non_linearity : function
                    The activation function of the layer applied to the feature vectors element-wise
                readout_aggregator : function
                    The aggregation function for all the node features for the final score prediction
                target_score_type : int
                    The output dimension of the GAT NN: predicting some of the Big-Five personality scores
                use_batch_norm : Bool
                    Flag for using Batch Normalization
                decay_rate : float
                    Rate of decay for the L2 and robustness regularization losses

                Returns
                -------
                model : Keras Model built on the Functional API
                model_loss() : Custom loss of MSE + robustness regularization

        """
        # Define the input placeholders into which data will be fed (batch shape included by default)
        batch_node_features = Input(shape=(dim_nodes, dim_feats), name='node_features')
        batch_adj_mats = Input(shape=(dim_nodes, dim_nodes), name='adjacency_matrices')
        batch_bias_mats = Input(shape=(dim_nodes, dim_nodes), name='masks')

        # integrate the hyper-parameters of the arch. depending on the type of READOUT used
        mutable_hu = hidden_units.copy()
        mutable_ah = attention_heads.copy()
        layer_args = {'dropout_rate': attn_drop,
                      'flag_batch_norm': use_batch_norm,
                      'decay_rate': decay_rate,
                      'flag_edge_weights': include_ew}
        master_heads = master_feats = 0
        if readout_aggregator is TensorflowGraphGAT.master_node_aggregator:
            master_heads = mutable_ah.pop()
            master_feats = mutable_hu.pop()

        # input GAT layer
        layer_args.update({'F_': mutable_hu[0],
                           'attn_heads': mutable_ah[0],
                           'attn_heads_reduction': 'concat',
                           'activation': non_linearity})
        input_layer = GATLayer(**layer_args)([batch_node_features, batch_adj_mats, batch_bias_mats])
        # hidden GAT layers
        prev_out = input_layer
        for i in range(1, len(mutable_ah) - 1):
            layer_args.update({'F_': mutable_hu[i],
                               'attn_heads': mutable_ah[i],
                               'attn_heads_reduction': 'concat',
                               'activation': non_linearity})
            i_th_layer = GATLayer(**layer_args)(
                inputs=[prev_out, batch_adj_mats, batch_bias_mats])
            prev_out = i_th_layer
            # accumulate the regularization losses

        # output GAT layer
        layer_args.update({'F_': mutable_hu[-1],
                           'attn_heads': mutable_ah[-1]})
        # choose the activation of the last layer and attention heads aggregation for specific readout
        if readout_aggregator is TensorflowGraphGAT.master_node_aggregator:
            layer_args.update({'attn_heads_reduction': 'concat',
                               'activation': non_linearity})
        else:
            layer_args.update({'flag_batch_norm': False,
                               'attn_heads_reduction': 'average',
                               'activation': lambda x: x})
        last_layer = GATLayer(**layer_args)(
            inputs=[prev_out, batch_adj_mats, batch_bias_mats])

        # average the regularization losses by the total nr of attention heads

        # aggregate all the output node features using the specified strategy
        gat_output = readout_aggregator(model_gat_output=last_layer, target_score_type=target_score_type,
                                        decay_rate=decay_rate, dim_nodes=dim_nodes, attn_drop=attn_drop,
                                        master_heads=master_heads, master_feats=master_feats)
        # define the Keras Model based on the Functional API
        model = Model(outputs=[gat_output], inputs=[batch_node_features, batch_adj_mats, batch_bias_mats])
        model.summary()

        return model
