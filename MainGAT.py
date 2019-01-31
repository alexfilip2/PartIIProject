import numpy as np
import tensorflow as tf
import AttentionHead as attn_layer
from BaseGAT import BaseGAT
from ToolsFunctional import *
from ToolsStructural import *
from keras_implementation.KerasAttentionHead import GraphAttention

dense = tf.layers.dense
checkpts_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'GAT_checkpoints')
if not os.path.exists(checkpts_dir):
    os.makedirs(checkpts_dir)


class MainGAT(BaseGAT):

    def average_feature_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop):
        # sum all the node features vectors into one of length target_score_type
        # shape of this is [[x_1,x_2,x_3,...,x_hid_units[-1]]]
        out_avg = tf.reduce_sum(model_GAT_output, axis=1)
        # fed them into a MLP with separate weights for each personality score
        # output = tf.layers.dense(output, units=target_score_type, use_bias=False)
        return out_avg

    def concat_feature_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop):
        # concatenate all the node features
        # shape of this is [[x_1,x_2,...,x_(n_nodes*hid_units[-1])]
        out_concat = tf.reshape(model_GAT_output, [1, -1])
        out = tf.layers.dense(out_concat, units=target_score_type, use_bias=True)
        return out

    def master_node_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop, master_heads=3):
        # model_GAT_output: (1,N,F')
        init_nb_nodes = int(model_GAT_output.shape[1])
        master_feats = np.zeros((1, 1, model_GAT_output.shape[-1]))
        extended_feats = tf.concat([model_GAT_output, master_feats], axis=1)
        # extended_feats: (1,N+1,F')
        adj_mat, bias_mat = attach_master(model_GAT_output.shape[1])

        adj_mat = tf.squeeze(adj_mat, axis=0)  # (N+1,N+1)
        bias_mat = tf.to_float(tf.squeeze(bias_mat, axis=0))  # (N+1,N+1)

        extended_feats = tf.squeeze(extended_feats, axis=0)

        # master GAT layer
        input_layer = GraphAttention(F_=target_score_type,
                                     attn_heads=master_heads,
                                     attn_heads_reduction='average',
                                     dropout_rate=attn_drop,
                                     activation=lambda x: x)
        feats_len_input = int(extended_feats.shape[-1])
        input_layer.build(F=feats_len_input)
        out, _, _ = input_layer.call(inputs=[extended_feats, adj_mat, bias_mat, tf.constant(False)])
        # take the resulted features of the master node
        out = tf.slice(input_=out, begin=[init_nb_nodes, 0], size=[1, target_score_type])

        return out

    def inference(self, in_feat_vects, adj_mat, bias_mat, hid_units, n_heads,
                  attn_drop, ffd_drop, activation=tf.nn.elu, residual=False,
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
        attn_heads_uloss = []
        attn_heads_eloss = []
        # for the first layer we provide the inputs directly
        for _ in range(n_heads[0]):
            attn_head_out, attn_unif_loss, attn_excl_loss = attn_layer.attn_head(input_feat_seq=in_feat_vects,
                                                                                 out_size=hid_units[0],
                                                                                 adj_mat=adj_mat,
                                                                                 bias_mat=bias_mat,
                                                                                 activation=activation,
                                                                                 include_weights=include_weights,
                                                                                 input_drop=ffd_drop,
                                                                                 coefficient_drop=attn_drop,
                                                                                 residual=False)
            attns.append(attn_head_out)
            attn_heads_uloss.append(attn_unif_loss)
            attn_heads_eloss.append(attn_excl_loss)

        # foe each node j we concatenate hj' obtained from each attention head, length of h_1 is still nr of nodes
        h_1 = tf.concat(attns, axis=-1)
        # now we have a new set of hi for the next layer in the same format with different dimensionality
        # sequence together all the len(hid_units) layers of stacked attention heads
        for i in range(1, len(hid_units) - 1):
            attns = []
            for _ in range(n_heads[i]):
                attn_head_out, attn_unif_loss, attn_excl_loss = attn_layer.attn_head(input_feat_seq=h_1,
                                                                                     out_size=hid_units[i],
                                                                                     adj_mat=adj_mat,
                                                                                     bias_mat=bias_mat,
                                                                                     activation=activation,
                                                                                     include_weights=include_weights,
                                                                                     input_drop=ffd_drop,
                                                                                     coefficient_drop=attn_drop,
                                                                                     residual=residual)
                attns.append(attn_head_out)
                attn_heads_uloss.append(attn_unif_loss)
                attn_heads_eloss.append(attn_excl_loss)

            # this is the input tensor for the next layer
            h_1 = tf.concat(attns, axis=-1)

        out = []
        # the output layer of the neural network architecture (node classification is implemented here)
        # the F' is nb_classes

        if aggregator is MainGAT.average_feature_aggregator:
            hid_units[-1] = target_score_type
        for i in range(n_heads[-1]):
            attn_head_out, attn_unif_loss, attn_excl_loss = attn_layer.attn_head(input_feat_seq=h_1,
                                                                                 out_size=hid_units[-1],
                                                                                 adj_mat=adj_mat,
                                                                                 bias_mat=bias_mat,
                                                                                 activation=lambda x: x,
                                                                                 include_weights=include_weights,
                                                                                 input_drop=ffd_drop,
                                                                                 coefficient_drop=attn_drop,
                                                                                 residual=False)
            out.append(attn_head_out)
            attn_heads_uloss.append(attn_unif_loss)
            attn_heads_eloss.append(attn_excl_loss)

        # average the outputs of the output attention heads for the final prediction (concatenation is not possible)
        model_GAT_output = tf.add_n(out) / n_heads[-1]
        # aggregate all the output node features
        output = aggregator(self, model_GAT_output=model_GAT_output, target_score_type=target_score_type,
                            attn_drop=attn_drop, ffd_drop=ffd_drop)
        # aggregate all the attention head losses
        model_unif_loss = tf.reduce_mean(attn_heads_uloss, axis=-1)
        model_excl_loss = tf.reduce_mean(attn_heads_eloss, axis=-1)

        print('Shape of the embedding output of the neural network for an input graph is ' + str(output.shape))
        return output, model_unif_loss, model_excl_loss

    def inference_keras(self, in_feat_vects, adj_mat, bias_mat, hid_units, n_heads,
                        attn_drop, ffd_drop, include_weights, activation=tf.nn.elu, residual=False
                        , aggregator=concat_feature_aggregator, target_score_type=5):
        adj_mat = tf.squeeze(adj_mat, axis=0)
        bias_mat = tf.squeeze(bias_mat, axis=0)
        in_feat_vects = tf.squeeze(in_feat_vects, axis=0)
        # change the length of the final features produced if just plain averaging is used
        if aggregator is MainGAT.average_feature_aggregator:
            hid_units[-1] = target_score_type

        # input GAT layer
        input_layer = GraphAttention(F_=hid_units[0],
                                     attn_heads=n_heads[0],
                                     attn_heads_reduction='concat',
                                     dropout_rate=attn_drop,
                                     activation=activation)
        feats_len_input = int(in_feat_vects.shape[-1])
        input_layer.build(F=feats_len_input)
        out, arch_uloss, arch_eloss = input_layer.call(inputs=[in_feat_vects, adj_mat, bias_mat, include_weights])

        # hidden GAT layers
        for i in range(1, len(n_heads) - 1):
            i_th_layer = GraphAttention(F_=hid_units[i],
                                        attn_heads=n_heads[i],
                                        attn_heads_reduction='concat',
                                        dropout_rate=attn_drop,
                                        activation=activation)
            feats_len_input = int(out.shape[-1])
            i_th_layer.build(F=feats_len_input)
            out, layer_uloss, layer_eloss = i_th_layer.call(inputs=[out, adj_mat, bias_mat, include_weights])
            arch_uloss = tf.add(arch_uloss, layer_uloss)
            arch_eloss = tf.add(arch_eloss, layer_eloss)

        # output GAT layer
        output_layer = GraphAttention(F_=hid_units[-1],
                                      attn_heads=n_heads[-1],
                                      attn_heads_reduction='average',
                                      dropout_rate=attn_drop,
                                      activation=lambda x: x)
        feats_len_input = int(out.shape[-1])
        output_layer.build(F=feats_len_input)
        gat_output, output_uloss, output_eloss = output_layer.call(inputs=[out, adj_mat, bias_mat, include_weights])

        arch_uloss = tf.divide(tf.add(arch_uloss, output_uloss), np.array(n_heads).sum())
        arch_eloss = tf.divide(tf.add(arch_eloss, output_eloss), np.array(n_heads).sum())

        # aggregate all the output node features
        output = aggregator(self, model_GAT_output=tf.expand_dims(gat_output, axis=0),
                            target_score_type=target_score_type,
                            attn_drop=attn_drop,
                            ffd_drop=ffd_drop)

        return output, arch_uloss, arch_eloss


# class embodying the hyperparameter choice of a GAT model
class GAT_hyperparam_config(object):

    def __init__(self, updated_params=None):
        self.params = {
            'hidden_units': [20, 40, 20],
            'attention_heads': [5, 5, 4],
            'include_ew': True,
            'readout_aggregator': MainGAT.master_node_aggregator,
            'num_epochs': 10000,
            'load_specific_data': load_struct_data,
            'pers_traits_selection': ['NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'],
            'batch_size': 2,
            'edgeWeights_filter': None,
            'patience': 50,
            'CHECKPT_PERIOD': 25,
            'learning_rate': 0.0001,
            'l2_coefficient': 0.0005,
            'residual': False,
            'attn_drop': 0.0,
            'ffd_drop': 0.0,
            'non_linearity': tf.nn.elu,
            'random_seed': 123,
            'eval_fold_in': 1,
            'eval_fold_out': 4,
            'k_outer': 5,
            'k_inner': 5,
            'nested_CV_level': 'outer'

        }
        self.update(update_hyper=updated_params)
        if self.params['nested_CV_level'] not in {'inner', 'outer'}:
            raise ValueError('Possbile CV levels: inner, outer')

    def __str__(self):
        str_traits = 'PT_' + "".join([pers.split('NEO.NEOFAC_')[1] for pers in self.params['pers_traits_selection']])
        str_attn_heads = 'AH_' + ",".join(map(str, self.params['attention_heads']))
        str_hid_units = 'HU_' + ",".join(map(str, self.params['hidden_units']))
        str_aggregator = 'AGR_' + self.params['readout_aggregator'].__name__.split('_')[0]
        str_limits = 'EL_' + ('None' if self.params['edgeWeights_filter'] is None else str(self.params['ew_limits']))
        str_batch_sz = '_BS_' + str(self.params['batch_size'])
        str_dataset = 'GAT_' + self.params['load_specific_data'].__name__.split('_')[1]
        str_include_ew = 'IW_' + str(self.params['include_ew'])
        str_cross_val = 'CV_' + str(self.params['eval_fold_in']) + str(self.params['eval_fold_out']) + self.params[
            'nested_CV_level']

        return '_'.join([str_dataset, str_attn_heads, str_hid_units, str_traits, str_aggregator, str_include_ew,
                         str_limits, str_batch_sz, str_cross_val])

    def update(self, update_hyper):
        if update_hyper is not None:
            self.params.update(update_hyper)

    def checkpt_file(self):
        return os.path.join(checkpts_dir, 'checkpoint_' + str(self))

    def logs_file(self):
        return os.path.join(checkpts_dir, 'logs_' + str(self))
