from __future__ import absolute_import
import tensorflow as tf
from keras import activations, constraints, initializers, regularizers
from keras import backend as K

from keras.layers import Layer, LeakyReLU, Multiply, Dropout, BatchNormalization


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.4,
                 decay_rate=0.0005,
                 activation=activations.relu,
                 flag_batch_norm=True,
                 flag_include_ew=True,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2,
                 bias_regularizer=regularizers.l2,
                 attn_kernel_regularizer=regularizers.l2,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Dimensionality of new node features: (F' in the paper
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activation  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.use_batch_norm = flag_batch_norm
        self.use_ew = flag_include_ew
        self.decay_rate = decay_rate

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = kernel_regularizer(decay_rate)
        self.bias_regularizer = bias_regularizer(decay_rate)
        self.attn_kernel_regularizer = attn_kernel_regularizer(decay_rate)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.biases = []  # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_
        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head), )
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs, **kwargs):
        input_node_feats, adjacency_mat, attn_mask = inputs
        # input_node_feats: Node features (? x N x F)
        # adjacency_mat: Adjacency matrix (? x N x N)
        # attn_mask : Bias matrix (? x N x N)
        # is_train: bool - specifies if dropout is active/inactive
        outputs, layer_u_loss, layer_e_loss = [], [], []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(input_node_feats, kernel)  # (? x N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])  # (? x N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (? x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs, perm=[0, 2, 1])  # (? x N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            dense += attn_mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (? x N x N)
            # Include edge weights
            if self.use_ew:
                dense = Multiply()([dense, adjacency_mat])

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(rate=self.dropout_rate)(inputs=dense)
            dropout_feat = Dropout(rate=self.dropout_rate)(inputs=features)

            # Linear combination with neighbors' features
            node_features = K.batch_dot(dropout_attn, dropout_feat)  # (? x N x F')
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # calculate how many neighbours of each node contribute to the aggregation (non-zero alpha)
            pos_alpha_coeffs = tf.count_nonzero(dense, axis=-1)

            # calculate the number of neighbours of each node
            node_deg = tf.count_nonzero(adjacency_mat, axis=-1)

            # the UNIFORM LOSS of the current attention head
            uniformity_loss = tf.reduce_mean(tf.to_float(tf.subtract(pos_alpha_coeffs, node_deg)), axis=-1)  # (?,)

            # the EXCLUSIVE LOSS of the current attention head
            exclusivity_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(dense), axis=-1), axis=-1)  # (?,)
            # Add output of attention head to final output
            outputs.append(node_features)
            layer_u_loss.append(uniformity_loss)
            layer_e_loss.append(exclusivity_loss)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        # apply batch normalization
        if self.use_batch_norm:
            batch_norm_features = BatchNormalization(axis=-1,
                                                     beta_regularizer=regularizers.l2(self.decay_rate),
                                                     gamma_regularizer=regularizers.l2(self.decay_rate))(inputs=output)
        else:
            batch_norm_features = output

        # apply activation
        activation_out = self.activation(batch_norm_features)
        # total u-loss for all the heads of the layer for ? input graphs
        layer_u_loss = K.sum(layer_u_loss, axis=0)
        # total e-loss for all the heads of the layer for ? input graphs
        layer_e_loss = K.sum(layer_e_loss, axis=0)

        return [activation_out, layer_u_loss, layer_e_loss]

    def compute_output_shape(self, input_shape):
        output_shape = [(input_shape[0][0], input_shape[0][1], self.output_dim), (), ()]
        return output_shape
