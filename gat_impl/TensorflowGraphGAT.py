from gat_impl.KerasAttentionHead import *
from utils.ToolsDataProcessing import attach_master
import numpy as np
from keras.layers import Input, Lambda, Dense
from keras.models import Model


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

        def readuce_sum(layer):
            return tf.reduce_sum(layer, axis=1)

        in_mlp = Lambda(function=readuce_sum)(model_GAT_output)
        out_avg = Dense(units=kwargs['target_score_type'],
                        kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                        use_bias=False)(in_mlp)
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
        def concat(input):
            dyn_shape = input.get_shape().as_list()
            concat_axis = np.prod(dyn_shape[1:])
            return tf.reshape(input, [-1, concat_axis])

        in_mlp = Lambda(function=concat)(model_GAT_output)
        # fed the result into MLP layer with separate weights for each personality score
        out_concat = Dense(units=kwargs['target_score_type'],
                           kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                           use_bias=False)(in_mlp)
        return out_concat

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
        # connect the master to all the nodes and  pairwise disconnect these
        extended_adj, extended_bias = attach_master(kwargs['dim_nodes'])

        def prepare_adjs(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_adj, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        def prepare_biases(gat_output):
            # repeat the same adjacency matrix for all the graphs in the batch
            return tf.cast(tf.tile(extended_bias, [tf.shape(gat_output)[0], 1, 1]), dtype=tf.float32)

        batch_ext_adjs = Lambda(function=prepare_adjs)(model_GAT_output)
        batch_ext_bias = Lambda(function=prepare_biases)(model_GAT_output)

        # master GAT layer
        master_layer, _, _ = GraphAttention(F_=kwargs['master_feats'],
                                            attn_heads=kwargs['master_heads'],
                                            attn_heads_reduction='average',
                                            flag_batch_norm=False,
                                            flag_include_ew=False,
                                            dropout_rate=kwargs['attn_drop'],
                                            decay_rate=kwargs['decay_rate'],
                                            activation=lambda x: x)(
            inputs=[model_GAT_output, batch_ext_adjs, batch_ext_bias])

        # extract only the high-level features produced for the master node (in each batch graph)
        def extract_master_feats(inputs):
            return tf.squeeze(tf.slice(input_=inputs, begin=[0, kwargs['dim_nodes'] - 1, 0],
                                       size=[tf.shape(inputs)[0], 1, kwargs['master_feats']]), axis=1)

        in_mlp = Lambda(function=extract_master_feats)(master_layer)

        out_master = Dense(units=kwargs['target_score_type'],
                           kernel_regularizer=regularizers.l2(kwargs['decay_rate']),
                           use_bias=False)(in_mlp)

        return out_master

    def inference_keras(self, dim_nodes, dim_feats, hidden_units, attention_heads,
                        attn_drop, decay_rate, use_batch_norm, include_ew, non_linearity, readout_aggregator,
                        target_score_type,
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
        batch_node_features = Input(shape=(dim_nodes, dim_feats))
        batch_adj_mats = Input(shape=(dim_nodes, dim_nodes))
        batch_bias_mats = Input(shape=(dim_nodes, dim_nodes))
        # change the dimension of the final features produced if averaging is employed
        mutable_hu = hidden_units.copy()
        mutable_ah = attention_heads.copy()
        layer_args = {'dropout_rate': attn_drop,
                      'flag_batch_norm': use_batch_norm,
                      'decay_rate': decay_rate,
                      'flag_include_ew': include_ew
                      }
        master_heads = master_feats = 0
        if readout_aggregator is TensorflowGraphGAT.master_node_aggregator:
            master_heads = mutable_ah.pop()
            master_feats = mutable_hu.pop()

        # input GAT layer
        layer_args.update({'F_': mutable_hu[0],
                           'attn_heads': mutable_ah[0],
                           'attn_heads_reduction': 'concat',
                           'activation': non_linearity, })
        input_layer, model_u_loss, model_e_loss = GraphAttention(**layer_args)(
            inputs=[batch_node_features, batch_adj_mats, batch_bias_mats])
        prev_out = input_layer

        # hidden GAT layers
        for i in range(1, len(mutable_ah) - 1):
            layer_args.update({'F_': mutable_hu[i],
                               'attn_heads': mutable_ah[i],
                               'attn_heads_reduction': 'concat',
                               'activation': non_linearity})
            i_th_layer, layer_u_loss, layer_e_loss = GraphAttention(**layer_args)(
                inputs=[prev_out, batch_adj_mats, batch_bias_mats])
            prev_out = i_th_layer
            # accumulate the regularization losses
            model_u_loss = tf.add_n([model_u_loss, layer_u_loss])
            model_e_loss = tf.add_n([model_e_loss, layer_e_loss])

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

        last_layer, last_u_loss, last_e_loss = GraphAttention(**layer_args)(
            inputs=[prev_out, batch_adj_mats, batch_bias_mats])

        # average the regularization losses by the total nr of attention heads
        nb_attn_heads = np.sum(np.array(mutable_ah))
        model_u_loss = tf.divide(tf.add(model_u_loss, last_u_loss), nb_attn_heads)
        model_e_loss = tf.divide(tf.add(model_e_loss, last_e_loss), nb_attn_heads)

        # aggregate all the output node features using the specified strategy
        gat_output = readout_aggregator(self, model_GAT_output=last_layer, target_score_type=target_score_type,
                                        decay_rate=decay_rate, dim_nodes=dim_nodes,
                                        attn_drop=attn_drop, master_heads=master_heads, master_feats=master_feats)
        model = Model(outputs=[gat_output],
                      inputs=[batch_node_features, batch_adj_mats, batch_bias_mats])
        print(model.summary())

        def model_loss():
            """" Wrapper function which calculates auxiliary values for the complete loss function.
             Returns a *function* which calculates the complete loss given only the input and target output """

            def full_mse_loss(y_true, y_pred):
                mse_loss = K.mean(K.square(y_pred - y_true))
                robustness_loss = tf.add_n([model_u_loss * decay_rate, model_e_loss * decay_rate])
                robustness_loss = K.mean(robustness_loss)
                """ Final loss calculation function to be passed to optimizer"""
                return mse_loss + robustness_loss

            return full_mse_loss

        return model, model_loss()
