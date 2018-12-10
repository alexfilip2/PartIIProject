import tensorflow as tf

conv1d = tf.layers.conv1d

'''
# the attention mechanism
# input_feat_seq - a tensor of subtensors of length F: the feature vectors of all the nodes in an example graph
# out_size is F' specified in the paper: the new length of the node feat. vectors.
# bias_mat is not W, but a form of adjancency matrix used to discard e_ij when i and j are not treated as neighbours 
(could be more than one-hop)
# adj_mat: is the initial graph with negative edge-weights filtered out which is used to weight the attention coeffs 
with the edge weight
'''


def attn_head(input_feat_seq, out_size, adj_mat, bias_mat, activation, input_drop=0.0, coefficient_drop=0.0,
              residual=False):
    # name_scope is a context manager which adds the operations to the computation graph
    with tf.name_scope('my_attn'):
        # apply dropout to the input units of the layer
        if input_drop != 0.0:
            input_feat_seq = tf.nn.dropout(input_feat_seq, 1.0 - input_drop)

        # the 1 in conv1d is the length of the 1D convolution window: apply the kernel to each elem of input_feat_seq
        # this conv1D layer is the linear transformation using the learnable matrix W (in the paper)
        seq_fts = tf.layers.conv1d(input_feat_seq, out_size, 1, use_bias=False)
        # now seq_fts is W*h_j for all nodes j

        # simplest self-attention possible
        # we apply the learnable weight vector a of length (2*F') to all pairs of  new feat.vec of length F'
        # applies the first half of a, outputs a real value (partial dot product) for each feat.vec. in seq_fts
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # applies the second half of a
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        # logits - the matrix of the e_ij attention coefficients e_ij is different from e_ji between all the nodes
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # add the edge weights of the graph to the e_ij coefficients
        logits = tf.matmul(logits, adj_mat)
        # compute the final coefficients alpha_ij (in the paper) by applying ReLu
        alpha_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coefficient_drop != 0.0:
            alpha_coefs = tf.nn.dropout(alpha_coefs, 1.0 - coefficient_drop)
        if input_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - input_drop)

        # compute h_i', the new feat. vec. for node i obtained from the aggregation process
        vals = tf.matmul(alpha_coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        # TO DO: APPLY THRESHOLD ON THE INPUT GRAPHS WHEN THE EDGE WEIGHTS ARE NEGATIVE
        if residual:
            if input_feat_seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(input_feat_seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + input_feat_seq

        return activation(ret)  # activation of the attn head
