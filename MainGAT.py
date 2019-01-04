import numpy as np
import tensorflow as tf
import AttentionHead as attn_layer
from BaseGAT import BaseGAT
from ToolsFunctional import *
from ToolsStructural import *

dense = tf.layers.dense


def average_feature_aggregator(model_GAT_output, target_score_type):
    # sum all the node features
    logits = tf.reduce_mean(model_GAT_output, axis=-1)  # shape of this is [[x_1,x_2,x_3,...,x_hid_units[-1]]]
    # fed them into a MLP with separate weights for each personality score
    output = tf.layers.dense(logits, units=target_score_type, use_bias=True)
    return output


def concat_feature_aggregator(model_GAT_output, target_score_type):
    # concatenate all the node features
    logits = tf.reshape(model_GAT_output, [1, -1])  # shape of this is [[x_1,x_2,...,x_(n_nodes*hid_units[-1])]
    output = tf.layers.dense(logits, units=target_score_type, use_bias=True)
    return output


class MainGAT(BaseGAT):
    def inference(in_feat_vects, adj_mat, bias_mat, hid_units, n_heads,
                  train_flag, attn_drop, ffd_drop, activation=tf.nn.elu, residual=False,
                  include_weights=False, aggregator=concat_feature_aggregator, target_score_type=5):
        """  Links multiple layers of stacked GAT attention heads
            Parameters
            ----------
            in_feat_vects : tensor
                The node feature vectors for a single example graph
            adj_mat : tensor
                Adjacency matrix with edge weights
            bias_mat: tensor
                Bias matrix for masked attention
            n_heads: list of int
                Number of attention heads on each layer of the GAT NN
            hid_units: list of int
                The length of the node feature vecs produced by each attn head of each layer
            aggregator : function
                The aggregation function for all the node features following then a MLP
            target_score_type: int
                The output dimension of the GAT NN: predicting some of the Big-Five personality scores
            attn_drop, ffd_drop : float
            train_flag: bool

            Returns
            -------
            output: tensor of shape (1,target_score_type)
                The output of the whole model, which is a prediction for the input graph

        """

        attns = []
        # for the first layer we provide the inputs directly
        for _ in range(n_heads[0]):
            attns.append(attn_layer.attn_head(input_feat_seq=in_feat_vects,
                                              out_size=hid_units[0],
                                              adj_mat=adj_mat,
                                              bias_mat=bias_mat,
                                              activation=activation,
                                              include_weights=include_weights,
                                              input_drop=ffd_drop,
                                              coefficient_drop=attn_drop,
                                              residual=False))

        # foe each node j we concatenate hj' obtained from each attention head, length of h_1 is still nr of nodes
        h_1 = tf.concat(attns, axis=-1)
        # now we have a new set of hi for the next layer in the same format with different dimensionality
        # sequence together all the len(hid_units) layers of stacked attention heads
        for i in range(1, len(hid_units) - 1):

            attns = []
            for _ in range(n_heads[i]):
                attns.append(attn_layer.attn_head(input_feat_seq=h_1,
                                                  out_size=hid_units[i],
                                                  adj_mat=adj_mat,
                                                  bias_mat=bias_mat,
                                                  activation=activation,
                                                  include_weights=include_weights,
                                                  input_drop=ffd_drop,
                                                  coefficient_drop=attn_drop,
                                                  residual=residual))

            # this is the input tensor for the next layer
            h_1 = tf.concat(attns, axis=-1)

        out = []
        # the output layer of the neural network architecture (node classification is implemented here)
        # the F' is nb_classes
        for i in range(n_heads[-1]):
            out.append(attn_layer.attn_head(input_feat_seq=h_1,
                                            out_size=hid_units[-1],
                                            adj_mat=adj_mat,
                                            bias_mat=bias_mat,
                                            activation=lambda x: x,
                                            include_weights=include_weights,
                                            input_drop=ffd_drop,
                                            coefficient_drop=attn_drop,
                                            residual=False))

        # average the outputs of the output attention heads for the final prediction (concatenation is not possible)
        model_GAT_output = tf.add_n(out) / n_heads[-1]
        # aggregate all the output node features
        output = aggregator(model_GAT_output=model_GAT_output, target_score_type=target_score_type)
        print('Shape of the embedding output of the neural network for an inpiut graph is ' + str(output.shape))
        return output


# class embodying the hyperparameter choice of a GAT model
class GAT_hyperparam_config(object):

    def __init__(self, hid_units, n_heads, nb_epochs, aggregator, include_weights, limits, batch_sz=1,
                 filter_name='interval',
                 pers_traits=None, dataset_type='struct', lr=0.0001, l2_coef=0.0005):
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.nb_epochs = nb_epochs
        self.aggregator = aggregator
        self.include_weights = include_weights
        self.batch_sz = batch_sz
        self.filter_name = filter_name
        if filter_name == 'interval':
            self.filter = interval_filter
            self.limits = limits
        else:
            self.filter = lambda x: x
            self.limits = []
        if pers_traits is None:
            self.pers_traits = ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E']
        else:
            self.pers_traits = ['NEO.NEOFAC_' + trait for trait in pers_traits]
        self.load_data = load_struct_data if dataset_type == 'struct' else load_funct_data
        self.dataset_type = dataset_type
        self.lr = lr
        self.l2_coef = l2_coef

    def __str__(self):
        str_traits = "".join([pers.split('NEO.NEOFAC_')[1] for pers in self.pers_traits])
        str_attn_heads = ",".join(map(str, self.n_heads))
        str_hid_units = ",".join(map(str, self.hid_units))
        str_aggregator = self.aggregator.__name__.split('_')[0]
        str_filter_limits = ",".join([str(int(x / 10000)) for x in self.limits])
        str_batch_sz = '' if self.batch_sz == 1 else '_BS_' + str(self.batch_sz)
        name = 'GAT_%s_AH%s_HU%s_PT_%s_AGR_%s_IW_%r_fltr_%s%s%s' % (self.dataset_type,
                                                                    str_attn_heads,
                                                                    str_hid_units,
                                                                    str_traits,
                                                                    str_aggregator,
                                                                    self.include_weights,
                                                                    self.filter_name,
                                                                    str_filter_limits,
                                                                    str_batch_sz)

        return name
