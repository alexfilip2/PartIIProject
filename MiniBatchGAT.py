import tensorflow as tf
from MainGAT import *

CHECKPT_PERIOD = 25
SETTLE_EPOCHS = 25

checkpts_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'GAT_checkpoints')
if not os.path.exists(checkpts_dir):
    os.makedirs(checkpts_dir)


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
            last_epoch_training = max(
                [int(ck_file.split('-')[-1]) for ck_file in ckpt.all_model_checkpoint_paths])
            print('Re-loading training from epoch %d' % last_epoch_training)
        # restart training from where it was left
        epoch_start = last_epoch_training + 1

    return epoch_start


def print_GAT_learn_loss(model_GAT_choice, tr_avg_loss, vl_avg_loss):
    train_losses_file = open(os.path.join(gat_model_stats, 'train_losses' + str(model_GAT_choice)), 'a')
    print('%.5f %.5f' % (tr_avg_loss, vl_avg_loss), file=train_losses_file)
    print('Training: loss = %.5f | Val: loss = %.5f' % (tr_avg_loss, vl_avg_loss))


def create_GAT_model(model_GAT_choice):
    # GAT model
    model = MainGAT
    # training hyper-parameters
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
    adj_matrices, graphs_features, score_train, score_test, score_val = model_GAT_choice.load_data(model_GAT_choice)
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
        saver = tf.train.Saver()
        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # minibatch operations
        # 0) Retrieve trainable variables
        tvs = tf.trainable_variables()
        # 1) Create placeholders for the accumulating gradients we'll be storing
        accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
        # 2) Operation to initialize accum_vars to zero
        zero_grads_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        # 3) Operation to compute the gradients for one minibatch
        # regularization loss of the parameters
        l2_l = tf.add_n([tf.nn.l2_loss(v) for v in tvs if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        gvs = opt.compute_gradients(loss + l2_l)
        # 4) Operation to accumulate the gradients in accum_vars
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        # 5) Operation to perform the update (apply gradients)
        apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(tf.trainable_variables())])

        # number of training graph examples
        tr_size = len(score_train)
        # number of validation examples
        vl_size = len(score_val)
        # number of test graph examples
        ts_size = len(score_test)
        print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))

        # Create session to execute ops
        sess = tf.InteractiveSession()
        # Necessary initializations
        tf.set_random_seed(1234)
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()).run()

        epoch_start = reload_GAT_model(model_GAT_choice=model_GAT_choice, sess=sess, saver=saver)
        constant_loss_tstamp = 0
        prev_tr_loss = sys.float_info.max

        # Train loop
        # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
        for epoch in range(epoch_start, nb_epochs + 1):
            # Array for logging the loss
            tr_loss_log = np.zeros(tr_size // batch_sz)
            # shuffle the training dataset
            score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = shuffle_tr_data(score_train,
                                                                            graphs_features,
                                                                            biases,
                                                                            adj_matrices,
                                                                            tr_size)
            for batch_nr in range(tr_size // batch_sz):
                # Make sure gradients are set to 0 before entering minibatch loop
                sess.run(zero_grads_ops)
                # Loop over minibatches and execute accumulate-gradient operation
                for batch_step in range(batch_sz):
                    index = batch_step + batch_nr * batch_sz
                    sess.run([accum_ops], feed_dict={
                        ftr_in: ftr_in_tr[index:index + 1],
                        bias_in: bias_in_tr[index:index + 1],
                        score_in: score_in_tr[index:index + 1],
                        adj_in: adj_in_tr[index:index + 1],
                        is_train: True,
                        attn_drop: 0.6,
                        ffd_drop: 0.6})
                # Done looping over minibatches. Now apply gradients.
                sess.run(apply_ops)

                for batch_step in range(batch_sz):
                    index = batch_step + batch_nr * batch_sz
                    (tr_example_loss,) = sess.run([loss], feed_dict={
                        ftr_in: ftr_in_tr[index:index + 1],
                        bias_in: bias_in_tr[index:index + 1],
                        score_in: score_in_tr[index:index + 1],
                        adj_in: adj_in_tr[index:index + 1],
                        is_train: True,
                        attn_drop: 0.6,
                        ffd_drop: 0.6})
                    tr_loss_log[batch_nr] += tr_example_loss
                tr_loss_log[batch_nr] /= batch_sz

            vl_avg_loss = 0
            for vl_step in range(tr_size, tr_size + vl_size):
                (vl_example_loss,) = sess.run([loss], feed_dict={
                    ftr_in: graphs_features[vl_step:vl_step + 1],
                    bias_in: biases[vl_step:vl_step + 1],
                    score_in: score_val[vl_step - tr_size:vl_step - tr_size + 1],
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

            # wait for the learning losses to settle before the specified number of training iterations
            THRESHOLD = 2.0
            if abs(tr_avg_loss - vl_avg_loss) < THRESHOLD and abs(tr_avg_loss - prev_tr_loss) < THRESHOLD:
                constant_loss_tstamp += 1
            else:
                constant_loss_tstamp = 0
            # if suffiecient epochs have passed and the losses are not changing finish the training
            if constant_loss_tstamp == SETTLE_EPOCHS: break
            prev_tr_loss = tr_avg_loss

        model_file = os.path.join(current_chkpt_dir, 'trained_model')
        if not tf.train.checkpoint_exists(model_file):
            save_path = saver.save(sess, model_file)
            print("Fully trained model saved in path: %s" % save_path)

        # restoring a pre-trained model
        saver.restore(sess, model_file)
        print("Model restored.")

        ts_avg_loss = 0
        for ts_step in range(tr_size + vl_size, tr_size + vl_size + ts_size):
            (ts_example_loss,) = sess.run([loss], feed_dict={
                ftr_in: graphs_features[ts_step:ts_step + 1],
                bias_in: biases[ts_step:ts_step + 1],
                score_in: score_test[ts_step - tr_size - vl_size:ts_step - tr_size - vl_size + 1],
                adj_in: adj_matrices[ts_step:ts_step + 1],
                is_train: False,
                attn_drop: 0.0,
                ffd_drop: 0.0})
            ts_avg_loss += ts_example_loss

        print('Test loss:', ts_avg_loss / ts_size)

        sess.close()


if __name__ == "__main__":
    hid_units = [64, 32, 16]
    n_heads = [4, 4, 6]
    aggregators = [concat_feature_aggregator]
    include_weights = [True]
    limits = [(183, 263857)]
    pers_traits = [None, ['A']]
    batches = [1, 5, 10]
    for aggr, iw, limit, p_traits, batch_sz in product(aggregators, include_weights, limits, pers_traits, batches):
        model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                                 n_heads=n_heads,
                                                 nb_epochs=1500,
                                                 aggregator=aggr,
                                                 include_weights=iw,
                                                 filter_name='interval',
                                                 pers_traits=p_traits,
                                                 limits=limit,
                                                 batch_sz=batch_sz,
                                                 dataset_type='struct',
                                                 lr=0.0001,
                                                 l2_coef=0.0005)
        create_GAT_model(model_GAT_config)
