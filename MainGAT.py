import numpy as np
import tensorflow as tf
import AttentionHead as attn_layer
from BaseGAT import BaseGAT
from ToolsFunctional import *
from ToolsStructural import *

dense = tf.layers.dense
checkpts_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'GAT_checkpoints')
if not os.path.exists(checkpts_dir):
    os.makedirs(checkpts_dir)


class MainGAT(BaseGAT):

    def average_feature_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop):
        # sum all the node features
        logits = tf.reduce_mean(model_GAT_output, axis=-1)  # shape of this is [[x_1,x_2,x_3,...,x_hid_units[-1]]]
        # fed them into a MLP with separate weights for each personality score
        output = tf.layers.dense(logits, units=target_score_type, use_bias=True)
        return output

    def concat_feature_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop):
        # concatenate all the node features
        logits = tf.reshape(model_GAT_output, [1, -1])  # shape of this is [[x_1,x_2,...,x_(n_nodes*hid_units[-1])]
        output = tf.layers.dense(logits, units=target_score_type, use_bias=True)
        return output

    def master_node_aggregator(self, model_GAT_output, target_score_type, attn_drop, ffd_drop):
        adj_mat, bias_mat = attach_master(model_GAT_output.shape[1])
        master_feat = np.zeros((1, 1, model_GAT_output.shape[-1]))
        extended_feats = tf.concat([model_GAT_output, master_feat], axis=1)

        (out, _, _) = attn_layer.attn_head(input_feat_seq=extended_feats,
                                           out_size=target_score_type,
                                           adj_mat=adj_mat,
                                           bias_mat=bias_mat,
                                           activation= tf.nn.relu,
                                           include_weights=False,
                                           input_drop=ffd_drop,
                                           coefficient_drop=attn_drop,
                                           residual=False)
        out = tf.Print(out, [out], message="This is extended_feats: ", first_n=10, summarize=2000)
        output = tf.squeeze(
            tf.slice(input_=out, begin=[0, int(model_GAT_output.shape[1]), 0], size=[1, 1, target_score_type]), axis=0)
        output = tf.Print(output, [output], message="This is output vector: ", first_n=10, summarize=2000)
        return output

    def inference(self, in_feat_vects, adj_mat, bias_mat, hid_units, n_heads,
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
                            attn_drop=attn_drop,
                            ffd_drop=ffd_drop)
        # aggregate all the attention head losses
        model_unif_loss = tf.reduce_mean(attn_heads_uloss, axis=-1)
        model_excl_loss = tf.reduce_mean(attn_heads_eloss, axis=-1)

        print('Shape of the embedding output of the neural network for an input graph is ' + str(output.shape))
        return output, model_unif_loss, model_excl_loss


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


def reload_GAT_model(model_GAT_choice, sess, saver):
    # Checkpoint file for the training of the GAT model
    current_chkpt_dir = os.path.join(checkpts_dir, str(model_GAT_choice))
    model_file = os.path.join(current_chkpt_dir, 'trained_model')
    if not os.path.exists(current_chkpt_dir):
        os.makedirs(current_chkpt_dir)

    ckpt = tf.train.get_checkpoint_state(current_chkpt_dir)
    if ckpt is None:
        epoch_start = 1
    else:
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        if tf.train.checkpoint_exists(model_file):
            last_epoch_training = model_GAT_choice.nb_epochs
            print('Re-loading full model %s' % model_GAT_choice)
        else:
            last_epoch_training = max([int(ck_file.split('-')[-1]) for ck_file in ckpt.all_model_checkpoint_paths])
            print('Re-loading training from epoch %d' % last_epoch_training)
        # restart training from where it was left
        epoch_start = last_epoch_training + 1

    return epoch_start


def print_GAT_learn_loss(model_GAT_choice, tr_avg_loss, vl_avg_loss):
    train_losses_file = open(os.path.join(gat_model_stats, 'train_losses' + str(model_GAT_choice)), 'a')
    print('%.5f %.5f' % (tr_avg_loss, vl_avg_loss), file=train_losses_file)
    print('Training: loss = %.5f | Val: loss = %.5f' % (tr_avg_loss, vl_avg_loss))
