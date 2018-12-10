import numpy as np
import tensorflow as tf
import AttentionHead as attn_layer
from BaseGAT import BaseGAT

dense = tf.layers.dense


# multiple layers of stacked GAT attention heads
# inputs - array of all the feature vectors of the nodes of a single input graph
# hid_units - numbers of hidden units (feat.vec.dim. output == F') per each attention head for each layer
# training - boolean value, doesn't do anything except for marking if training or testing
# n_heads - number of stacked attention heads for each layer of the architecture
class MainGAT(BaseGAT):
    def inference(in_feat_vects, train_flag, attn_drop, ffd_drop, adj_mat,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        # for the first layer we provide the inputs directly
        for _ in range(n_heads[0]):
            attns.append(attn_layer.attn_head(in_feat_vects, bias_mat=bias_mat, adj_mat=adj_mat,
                                              out_size=hid_units[0], activation=activation,
                                              input_drop=ffd_drop, coefficient_drop=attn_drop, residual=False))
        # foe each node j we concatenate hj' obtained from each attention head, length of h_1 is still nr of nodes
        h_1 = tf.concat(attns, axis=-1)
        # now we have a new set of hi for the next layer in the same format with different dimensionality
        # sequence together all the len(hid_units) layers of stacked attention heads
        for i in range(1, len(hid_units) - 1):

            attns = []
            for _ in range(n_heads[i]):
                attns.append(attn_layer.attn_head(h_1, bias_mat=bias_mat, adj_mat=adj_mat,
                                                  out_size=hid_units[i], activation=activation,
                                                  input_drop=ffd_drop, coefficient_drop=attn_drop, residual=residual))
            # this is the input tensor for the next layer
            h_1 = tf.concat(attns, axis=-1)

        out = []
        # the output layer of the neural network architecture (node classification is implemented here)
        # the F' is nb_classes
        for i in range(n_heads[-1]):
            out.append(attn_layer.attn_head(h_1, bias_mat=bias_mat, adj_mat=adj_mat,
                                            out_size=hid_units[-1], activation=lambda x: x,
                                            input_drop=ffd_drop, coefficient_drop=attn_drop, residual=False))
        # average the outputs of the output attention heads for the final prediction (concatenation is not possible)

        logits = tf.add_n(out) / n_heads[-1]
        # feed the output into a MLP in order to separate the weights for individual paeronality traits
        # aggregate the node features by averaging them
        output = tf.layers.dense(tf.reduce_mean(logits, axis=1), units=5, use_bias=True)
        # print(tf.reduce_mean(logits, axis=1).shape) === in the format [[x_1,x_2,x_3,...,x_hid_units[-1]]]

        print('Shape of the embedding output of the neural network for an inpiut graph is ' + str(output.shape))
        return output
