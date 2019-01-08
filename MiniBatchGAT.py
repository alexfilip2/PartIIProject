import tensorflow as tf
from MainGAT import *

CHECKPT_PERIOD = 25
patience = 100


def create_GAT_model(model_GAT_choice):
    # GAT model
    model = MainGAT
    # GAT hyper-parameters
    batch_sz = model_GAT_choice.batch_sz  # batch training size
    nb_epochs = model_GAT_choice.nb_epochs  # number of learning iterations over the trainign dataset
    lr = model_GAT_choice.lr  # learning rate
    l2_coef = model_GAT_choice.l2_coef  # weight decay
    hid_units = model_GAT_choice.hid_units  # numbers of features produced by each attention head per network layer
    n_heads = model_GAT_choice.n_heads  # number of attention heads on each layer
    residual = False
    nonlinearity = tf.nn.elu
    aggregator = model_GAT_choice.aggregator
    include_weights = model_GAT_choice.include_weights

    print('Name of the current GAT model is %s' % model_GAT_choice)
    print('Dataset: ' + model_GAT_choice.dataset_type + ' HCP graphs')
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    # data for adjancency matrices, node feature vectors and personality scores for each study patient
    adj_matrices, graphs_features, pers_scores = model_GAT_choice.load_data(model_GAT_choice)
    # used in order to implement MASKED ATTENTION by discardining non-neighbours out of nhood hops
    biases = adj_to_bias(adj_matrices, [graph.shape[0] for graph in adj_matrices], nhood=1)
    # nr of nodes for each graph: it is shared among all examples due to the dataset
    nb_nodes = adj_matrices[0].shape[0]
    # the initial length F of each node feature vector: for every graph, node feat.vecs. have the same length
    ft_size = graphs_features.shape[-1]
    # how many of the big-five personality traits the model is targeting at once
    outGAT_sz_target = len(model_GAT_choice.pers_traits)
    # checkpoint directory storing the progress of the current model
    current_chkpt_dir = os.path.join(checkpts_dir, str(model_GAT_choice))

    # create a TensofFlow session, the context of evaluation for the Tensor objects
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))
            bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
            score_in = tf.placeholder(dtype=tf.float32, shape=(1, outGAT_sz_target))
            adj_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        prediction = model.inference(in_feat_vects=ftr_in,
                                     adj_mat=adj_in,
                                     bias_mat=bias_in,
                                     hid_units=hid_units,
                                     n_heads=n_heads,
                                     target_score_type=outGAT_sz_target,
                                     train_flag=is_train,
                                     aggregator=aggregator,
                                     include_weights=include_weights,
                                     residual=residual,
                                     activation=nonlinearity,
                                     attn_drop=attn_drop,
                                     ffd_drop=ffd_drop)

        loss = tf.losses.mean_squared_error(labels=score_in, predictions=prediction)
        # create tf session saver
        saver = tf.train.Saver()

        # minibatch operations
        zero_grads_ops, accum_ops, apply_ops = model.batch_training(loss=loss, lr=lr, l2_coef=l2_coef)

        # number of training, validation, test graph examples
        split_sz = len(pers_scores) // 10
        tr_size, vl_size, ts_size = split_sz * 8, split_sz, split_sz
        print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))

        # Create interactive session to execute the accumulation of gradients per batch
        sess = tf.InteractiveSession()
        # Necessary initializations
        tf.set_random_seed(1234)
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()).run()

        # reload the GAT model from the last checkpoint
        epoch_start = reload_GAT_model(model_GAT_choice=model_GAT_choice, sess=sess, saver=saver)
        # record the minimum validation loss encountered until current epoch
        vlss_mn = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0

        # Train loop
        # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
        for epoch in range(epoch_start, nb_epochs + 1):
            # number of iterations of the training set when batch-training
            tr_iterations = tr_size // batch_sz
            # Array for logging the loss
            tr_loss_log = np.zeros(tr_iterations)

            # shuffle the training dataset
            score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = shuffle_tr_data(pers_scores,
                                                                            graphs_features,
                                                                            biases,
                                                                            adj_matrices,
                                                                            tr_size)
            for iteration in range(tr_iterations):
                # Make sure gradients are set to 0 before entering minibatch loop
                sess.run(zero_grads_ops)
                # Loop over minibatches and execute accumulate-gradient operation
                for batch_step in range(batch_sz):
                    index = batch_step + iteration * batch_sz
                    sess.run([accum_ops], feed_dict={ftr_in: ftr_in_tr[index:index + 1],
                                                     bias_in: bias_in_tr[index:index + 1],
                                                     score_in: score_in_tr[index:index + 1],
                                                     adj_in: adj_in_tr[index:index + 1],
                                                     is_train: True,
                                                     attn_drop: 0.6,
                                                     ffd_drop: 0.6})
                # Done looping over minibatches. Now apply gradients.
                sess.run(apply_ops)
                # Calculate the validation loss after every single batch training
                for batch_step in range(batch_sz):
                    index = batch_step + iteration * batch_sz
                    (tr_example_loss,) = sess.run([loss], feed_dict={ftr_in: ftr_in_tr[index:index + 1],
                                                                     bias_in: bias_in_tr[index:index + 1],
                                                                     score_in: score_in_tr[index:index + 1],
                                                                     adj_in: adj_in_tr[index:index + 1],
                                                                     is_train: True,
                                                                     attn_drop: 0.6,
                                                                     ffd_drop: 0.6})
                    tr_loss_log[iteration] += tr_example_loss
                tr_loss_log[iteration] /= batch_sz

            vl_avg_loss = 0
            for vl_step in range(tr_size, tr_size + vl_size):
                (vl_example_loss,) = sess.run([loss], feed_dict={ftr_in: graphs_features[vl_step:vl_step + 1],
                                                                 bias_in: biases[vl_step:vl_step + 1],
                                                                 score_in: pers_scores[vl_step:vl_step + 1],
                                                                 adj_in: adj_matrices[vl_step:vl_step + 1],
                                                                 is_train: False,
                                                                 attn_drop: 0.0,
                                                                 ffd_drop: 0.0})
                vl_avg_loss += vl_example_loss
            vl_avg_loss /= vl_size

            tr_avg_loss = np.sum(tr_loss_log) / (tr_size // batch_sz)
            print_GAT_learn_loss(model_GAT_choice, tr_avg_loss=tr_avg_loss, vl_avg_loss=vl_avg_loss)

            checkpt_file = os.path.join(current_chkpt_dir, 'checkpoint')
            if epoch % CHECKPT_PERIOD == 0:
                save_path = saver.save(sess, checkpt_file, global_step=epoch)
                print("Training progress after %d epochs saved in path: %s" % (epoch, save_path))

            # wait for the validation loss to settle before the specified number of training iterations
            if vl_avg_loss <= vlss_mn:
                vlss_early_model = vl_avg_loss
                vlss_mn = np.min((vl_avg_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn)
                    print('Early stop model validation loss: ', vlss_early_model)
                    break

    model_file = os.path.join(current_chkpt_dir, 'trained_model')
    if not tf.train.checkpoint_exists(model_file):
        save_path = saver.save(sess, model_file)
        print("Fully trained model saved in path: %s" % save_path)

    # restoring a pre-trained model
    saver.restore(sess, model_file)
    print("Model restored.")

    ts_avg_loss = 0
    for ts_step in range(tr_size + vl_size, tr_size + vl_size + ts_size):
        (ts_example_loss,) = sess.run([loss], feed_dict={ftr_in: graphs_features[ts_step:ts_step + 1],
                                                         bias_in: biases[ts_step:ts_step + 1],
                                                         score_in: pers_scores[ts_step: ts_step + 1],
                                                         adj_in: adj_matrices[ts_step:ts_step + 1],
                                                         is_train: False,
                                                         attn_drop: 0.0,
                                                         ffd_drop: 0.0})
        ts_avg_loss += ts_example_loss

    print('Test loss:', ts_avg_loss / ts_size)

    sess.close()


if __name__ == "__main__":
    hid_units = [16, 8, 8]
    n_heads = [2, 3, 3]
    aggregators = [MainGAT.concat_feature_aggregator]
    include_weights = [True]
    limits = [(183, 263857)]
    pers_traits = [['A'], ['E'], ['N'], ['C'], ['O']]
    batches = [2]
    for aggr, iw, limit, p_traits, batch_size in product(aggregators, include_weights, limits, pers_traits, batches):
        model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                                 n_heads=n_heads,
                                                 nb_epochs=10000,
                                                 aggregator=aggr,
                                                 include_weights=iw,
                                                 filter_name='interval',
                                                 pers_traits=p_traits,
                                                 limits=limit,
                                                 batch_sz=batch_size,
                                                 dataset_type='struct',
                                                 lr=0.0001,
                                                 l2_coef=0.0005)
        create_GAT_model(model_GAT_config)
