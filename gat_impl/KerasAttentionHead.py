import tensorflow as tf
from keras import initializers, regularizers
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Multiply, Dropout, BatchNormalization


class GATLayer(Layer):
    def __init__(self,
                 F_,
                 attn_heads,
                 attn_heads_reduction,
                 dropout_rate,
                 decay_rate,
                 activation,
                 flag_batch_norm,
                 flag_edge_weights,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2,
                 bias_regularizer=regularizers.l2,
                 attn_kernel_regularizer=regularizers.l2,
                 **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average, not %s' % attn_heads_reduction)

        self.F_ = F_  # Dimensionality of the node features produced : F' in the GAT paper
        self.attn_heads = attn_heads  # Number of attention heads on this layer : K in the GAT paper
        self.attn_heads_reduction = attn_heads_reduction  # Reduction of attention heads  via average/concatenation
        self.dropout_rate = dropout_rate  # Dropout rate of alpha coefficients and input node features
        self.activation = activation  # Activation function of the layer
        self.use_bias = use_bias  # Use bias on the node features produced
        self.use_batch_norm = flag_batch_norm  # Use Batch Normalization on the output of this layer
        self.use_ew = flag_edge_weights  # Include edge weights into the learning process
        self.decay_rate = decay_rate  # L2 regularization coefficient
        # initializers for each type of parameters used by the layer
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        # regularizers for each type of parameters used by the layer
        self.kernel_regularizer = kernel_regularizer(decay_rate)
        self.bias_regularizer = bias_regularizer(decay_rate)
        self.attn_kernel_regularizer = attn_kernel_regularizer(decay_rate)

        # The entire set of parameters used by the layer, populated by build() - dependent on the input shape
        self.kernels = []  # Layer kernels for each attention head : W weight matrix in the GAT paper
        self.biases = []  # Layer biases for each attention head
        self.attn_kernels = []  # Attention kernels for each attention head: a weight vector in the GAT paper

        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads  # output shape: (? x N x KF')
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_  # output shape: (? x N x F')

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # Retrieve initial node feature dimension
        F = input_shape[0][-1]

        # Initialize the weights (kernels) and biases for each attention head
        for index_attn_head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     name='kernel_{}'.format(index_attn_head))
            self.kernels.append(kernel)
            # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       name='bias_{}'.format(index_attn_head))
                self.biases.append(bias)
            # Attention kernels for outgoing edges attention coefficients and ingoing: alpha_ij are not symmetric
            attn_kernel_outgoing = self.add_weight(shape=(self.F_, 1),
                                                   initializer=self.attn_kernel_initializer,
                                                   regularizer=self.attn_kernel_regularizer,
                                                   name='attn_kernel_self_{}'.format(index_attn_head))
            attn_kernel_ingoing = self.add_weight(shape=(self.F_, 1),
                                                  initializer=self.attn_kernel_initializer,
                                                  regularizer=self.attn_kernel_regularizer,
                                                  name='attn_kernel_neigh_{}'.format(index_attn_head))
            self.attn_kernels.append([attn_kernel_outgoing, attn_kernel_ingoing])

    def call(self, inputs, **kwargs) -> tf.Tensor:
        # input_node_feats: Batch Node features (? x N x F)
        # adjacency_mat:  Batch Weighted adjacency matrix (? x N x N)
        # attn_mask :  Batch bias matrices (? x N x N)
        input_node_feats, adjacency_mat, attn_mask = inputs

        # Lists of individual outputs, exclusivity and uniformity losses for each attention head
        outputs, layer_u_loss, layer_e_loss = [], [], []
        for index_attn_head in range(self.attn_heads):
            # Retrieve the parameters used by this specific head
            kernel = self.kernels[index_attn_head]
            attention_kernel = self.attn_kernels[index_attn_head]

            # Apply non-linear transformation W to the node features
            features = K.dot(input_node_feats, kernel)  # (? x N x F')

            # Apply the attention transformation for each pair of node features
            attn_outgoing = K.dot(features, attention_kernel[0])  # (? x N x 1)
            attn_ingoing = K.dot(features, attention_kernel[1])  # (? x N x 1)
            dense = attn_outgoing + tf.transpose(attn_ingoing, perm=[0, 2, 1])  # (? x N x N) via broadcasting of +

            # Add non-linearity
            dense = LeakyReLU(alpha=0.2)(dense)

            # Apply the mask to the attention coefficients
            dense += attn_mask
            # Apply Softmax to nullify the attention coefficients of not connected nodes
            dense = K.softmax(dense, axis=-1)  # (? x N x N)

            # Include edge weights by element-wise multiplication
            if self.use_ew:
                dense = Multiply()([dense, adjacency_mat])

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(rate=self.dropout_rate)(inputs=dense)
            dropout_feat = Dropout(rate=self.dropout_rate)(inputs=features)

            # Linear combination with neighbors' features - aggregation
            node_features = K.batch_dot(dropout_attn, dropout_feat, axes=[2, 1])  # (? x N x F')
            # Apply bias to the final node features produced by the attention mechanism
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[index_attn_head])

            # Calculate how many neighbours of each node contribute to the aggregation (positive attention coefficient)
            pos_attn_coeffs = tf.count_nonzero(dense, axis=-1)

            # Calculate the number of neighbours of each node
            node_deg = tf.count_nonzero(adjacency_mat, axis=-1)

            # Compute the uniformity loss of the current attention head
            uniformity_loss = tf.reduce_mean(tf.cast(tf.subtract(pos_attn_coeffs, node_deg), dtype=tf.float32),
                                             axis=-1)  # (?,)
            # Compute the exclusivity loss of the current attention head
            exclusivity_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(dense), axis=-1), axis=-1)  # (?,)

            # Stack the outputs
            outputs.append(node_features)
            layer_u_loss.append(uniformity_loss)
            layer_e_loss.append(exclusivity_loss)

        # Aggregate the attention heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x K*F')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # (N x F')

        # Apply batch normalization before the activation
        if self.use_batch_norm:
            batch_norm_features = BatchNormalization(axis=-1,
                                                     beta_regularizer=regularizers.l2(self.decay_rate),
                                                     gamma_regularizer=regularizers.l2(self.decay_rate))(inputs=output)
        else:
            batch_norm_features = output

        # total u-loss for all the heads of the layer for ? input graphs
        layer_u_loss = K.mean(K.mean(K.stack(layer_u_loss, axis=1), axis=-1))
        layer_u_loss = tf.identity(layer_u_loss, name="uniform_loss_{}".format(self.name))

        # total e-loss for all the heads of the layer for ? input graphs
        layer_e_loss = K.mean(K.mean(K.stack(layer_e_loss, axis=1), axis=-1))
        layer_e_loss = tf.identity(layer_e_loss, name="exclusive_loss_{}".format(self.name))

        # add the regularization losses to the overall loss
        self.add_loss(layer_u_loss * self.decay_rate)
        self.add_loss(layer_e_loss * self.decay_rate)

        # Apply activation on the high-level node features
        activation_out = self.activation(batch_norm_features)

        return activation_out

    def compute_output_shape(self, input_shape) -> list:
        output_shape = [(input_shape[0][0], input_shape[0][1], self.output_dim)]
        return output_shape
