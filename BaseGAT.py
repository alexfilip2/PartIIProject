import tensorflow as tf
from ToolsFunctional import *
from ToolsStructural import *
from keras_implementation.KerasAttentionHead import GraphAttention


class BaseGAT(object):

    def average_feature_aggregator(self, model_GAT_output, **kwargs):
        # sum all the node features vectors
        out_avg = tf.reduce_sum(model_GAT_output, axis=1)
        # fed them into a MLP with separate weights for each personality score
        # output = tf.layers.dense(output, units=target_score_type, use_bias=False)
        return out_avg

    def concat_feature_aggregator(self, model_GAT_output, **kwargs):
        # concatenate all the node features
        dyn_shape = model_GAT_output.get_shape().as_list()
        concat_dim = np.prod(dyn_shape[1:])
        out_concat = tf.reshape(model_GAT_output, [-1, concat_dim])

        # fed them into a MLP with separate weights for each personality score
        out = tf.layers.dense(out_concat, units=kwargs['target_score_type'], use_bias=True)
        return out

    def master_node_aggregator(self, model_GAT_output, **kwargs):
        # model_GAT_output: (?,N,F')
        init_nb_nodes = int(model_GAT_output.shape[1])
        dyn_batch_nr = tf.shape(model_GAT_output)[0]
        master_feats = tf.zeros([1, model_GAT_output.shape[-1]])
        master_feats_tilled = tf.expand_dims(tf.tile(master_feats, tf.stack([dyn_batch_nr, 1])), axis=1)
        extended_feats = tf.concat([model_GAT_output, master_feats_tilled], axis=1)  # extended_feats: (1,N+1,F')
        adj_mat, bias_mat = attach_master(init_nb_nodes)
        adj_mat = tf.cast(tf.tile(adj_mat, [dyn_batch_nr, 1, 1]), dtype=tf.float32)
        bias_mat = tf.cast(tf.tile(bias_mat, [dyn_batch_nr, 1, 1]), dtype=tf.float32)

        # master GAT layer
        master_layer, _, _ = GraphAttention(F_=kwargs['target_score_type'],
                                            attn_heads=kwargs['master_heads'],
                                            attn_heads_reduction='average',
                                            dropout_rate=kwargs['attn_drop'],
                                            activation=lambda x: x)(
            inputs=[extended_feats, adj_mat, bias_mat, kwargs['is_train']],include_ew=False)

        # take the resulted features of the master node
        expl_out = tf.slice(input_=master_layer, begin=[0, init_nb_nodes, 0],
                            size=[dyn_batch_nr, 1, kwargs['target_score_type']])

        return tf.squeeze(expl_out, axis=1)

    def inference_keras(self, in_feat_vects, adj_mat, bias_mat, hid_units, n_heads,
                        attn_drop, is_train, include_weights, activation=tf.nn.elu, residual=False
                        , aggregator=concat_feature_aggregator, target_score_type=5):

        # adj_mat, bias_mat, in_feat_vects = map(lambda x: tf.squeeze(x, axis=0), [adj_mat, bias_mat, in_feat_vects])
        # change the length of the final features produced if just plain averaging is used
        if aggregator is BaseGAT.average_feature_aggregator:
            hid_units[-1] = target_score_type
        # input GAT layer
        out_input_layer, arch_uloss, arch_eloss = GraphAttention(F_=hid_units[0],
                                                                 attn_heads=n_heads[0],
                                                                 attn_heads_reduction='concat',
                                                                 dropout_rate=attn_drop,
                                                                 activation=activation)(
            inputs=[in_feat_vects, adj_mat, bias_mat, is_train], include_ew=include_weights)

        # hidden GAT layers
        out = out_input_layer
        for i in range(1, len(n_heads) - 1):
            out_i_th_layer, layer_uloss, layer_eloss = GraphAttention(F_=hid_units[i],
                                                                      attn_heads=n_heads[i],
                                                                      attn_heads_reduction='concat',
                                                                      dropout_rate=attn_drop,
                                                                      activation=activation)(
                inputs=[out, adj_mat, bias_mat, is_train], include_ew=include_weights)
            out = out_i_th_layer
            arch_uloss = tf.add(arch_uloss, layer_uloss)
            arch_eloss = tf.add(arch_eloss, layer_eloss)

        # output GAT layer
        gat_output, output_uloss, output_eloss = GraphAttention(F_=hid_units[-1],
                                                                attn_heads=n_heads[-1],
                                                                attn_heads_reduction='average',
                                                                dropout_rate=attn_drop,
                                                                activation=lambda x: x)(
            inputs=[out, adj_mat, bias_mat, is_train],
            include_ew=include_weights)

        nb_attn_heads = np.sum(np.array(n_heads))
        arch_uloss = tf.divide(tf.add(arch_uloss, output_uloss), nb_attn_heads)
        arch_eloss = tf.divide(tf.add(arch_eloss, output_eloss), nb_attn_heads)

        # aggregate all the output node features
        output = aggregator(self, model_GAT_output=gat_output,
                            target_score_type=target_score_type, is_train=is_train,
                            attn_drop=attn_drop, master_heads=3)

        return output, arch_uloss, arch_eloss
