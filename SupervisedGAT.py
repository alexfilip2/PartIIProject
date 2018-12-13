import tensorflow as tf

from MainGAT import MainGAT
from ToolsFunctional import *
from ToolsStructural import *

CHECKPT_PERIOD = 25
checkpts_dir = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'GAT_checkpoints')
if not os.path.exists(checkpts_dir):
    os.makedirs(checkpts_dir)
gat_model_stats = os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'learning_process')
if not os.path.exists(gat_model_stats):
    os.makedirs(gat_model_stats)


def create_GAT_model(model_GAT_choice):
    # GAT model
    model = MainGAT
    # Checkpoint file for the training of the GAT model
    checkpt_file = os.path.join(checkpts_dir, str(model_GAT_choice))

    # training hyper-parameters
    batch_sz = 1  # batch training size; currently ONLY ONE example per training step: TO BE EXTENDED!!
    nb_epochs = model_GAT_choice.nb_epochs
    lr = model_GAT_choice.lr  # learning rate
    l2_coef = model_GAT_choice.l2_coef  # weight decay
    hid_units = model_GAT_choice.hid_units  # numbers of features produced by each attention head per network layer
    n_heads = model_GAT_choice.n_heads  # number of attention heads on each layer
    residual = False
    nonlinearity = tf.nn.elu

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

    # personality traits scores: 'NEO.NEOFAC_A', 'NEO.NEOFAC_O', 'NEO.NEOFAC_C', 'NEO.NEOFAC_N', 'NEO.NEOFAC_E'
    pers_traits = ['NEO.NEOFAC_' + trait for trait in model_GAT_choice.pers_traits]

    # data for adjancency matrices, node feature vectors and personality scores for each study patient
    load_data = load_struct_data if model_GAT_choice.dataset_type == 'structural' else load_funct_data
    adj_matrices, graphs_features, score_train, score_test, score_val = load_data(pers_traits,
                                                                                  model_GAT_choice.edge_w_limit)

    # used in order to implement MASKED ATTENTION by discardining non-neighbours out of nhood hops
    biases = adj_to_bias(adj_matrices, [graph.shape[0] for graph in adj_matrices], nhood=1)
    # nr of nodes for each graph: it is shared among all examples due to the dataset
    nb_nodes = adj_matrices[0].shape[0]
    # the initial length F of each node feature vector: for every graph, node feat.vecs. have the same length
    ft_size = graphs_features.shape[-1]
    # how many of the big-five personality traits the model is targeting at once
    outGAT_sz_target = len(pers_traits)

    # create a TensofFlow session, the context of evaluation for the Tensor objects
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, ft_size))
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
            score_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, outGAT_sz_target))
            adj_in = tf.placeholder(dtype=tf.float32, shape=(batch_sz, nb_nodes, nb_nodes))
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
                                     residual=residual,
                                     activation=nonlinearity,
                                     attn_drop=attn_drop,
                                     ffd_drop=ffd_drop)

        loss = tf.losses.mean_squared_error(labels=score_in, predictions=prediction)
        train_op = model.training(loss, lr, l2_coef)
        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # a Session object creates the environment in which Operations objects are executed and Tensor objects are evaluated
        with tf.Session() as sess:
            # number of training graph examples
            tr_size = len(score_train)
            # number of validation graph examples
            vl_size = len(score_val)
            # number of test graph examples
            ts_size = len(score_test)
            # restart training from where it was left
            epoch_start = 1

            print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))

            if not tf.train.checkpoint_exists(checkpt_file):
                print('The GAT model is not pre-trained, training it now...')
                partially_trained = False
                for epoch_step in reversed(range(nb_epochs)):
                    possible_last_checkpt = '%s-%d' % (checkpt_file, epoch_step)
                    if tf.train.checkpoint_exists(possible_last_checkpt):
                        # restoring a pre-trained model
                        epoch_start = epoch_step + 1
                        saver.restore(sess, possible_last_checkpt)
                        print("Model restored from epoch number %d" % epoch_step)
                        partially_trained = True
                        break

                if not partially_trained:
                    # run the initializer for the Variable objects to be optimized
                    open(os.path.join(gat_model_stats, 'train_losses_%s' % model_GAT_choice), 'w')
                    sess.run(init_op)

                # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
                for epoch in range(epoch_start, nb_epochs + 1):

                    # shuffle the training dataset
                    score_in_tr, ftr_in_tr, bias_in_tr, adj_in_tr = shuffle_tr_data(score_train,
                                                                                    graphs_features,
                                                                                    biases,
                                                                                    adj_matrices,
                                                                                    tr_size)
                    # training step
                    tr_step = 0
                    #  training loss of this epoch
                    train_loss_avg = 0
                    while tr_step < tr_size:
                        (_, loss_value_tr) = sess.run([train_op, loss],
                                                      feed_dict={
                                                          ftr_in: ftr_in_tr[tr_step:tr_step + 1],
                                                          bias_in: bias_in_tr[tr_step:tr_step + 1],
                                                          score_in: score_in_tr[tr_step:tr_step + 1],
                                                          adj_in: adj_in_tr[tr_step:tr_step + 1],
                                                          is_train: True,
                                                          attn_drop: 0.6,
                                                          ffd_drop: 0.6})
                        train_loss_avg += loss_value_tr
                        tr_step += 1

                    # validation step
                    vl_step = tr_size
                    # validation loss of this epoch
                    val_loss_avg = 0

                    while vl_step < vl_size + tr_size:
                        (loss_value_vl,) = sess.run([loss],
                                                    feed_dict={
                                                        ftr_in: graphs_features[vl_step:vl_step + 1],
                                                        bias_in: biases[vl_step:vl_step + 1],
                                                        score_in: score_val[vl_step - tr_size:vl_step + 1 - tr_size],
                                                        adj_in: adj_matrices[vl_step:vl_step + 1],
                                                        is_train: False,
                                                        attn_drop: 0.0,
                                                        ffd_drop: 0.0})

                        val_loss_avg += loss_value_vl
                        vl_step += 1
                    train_losses_file = open(os.path.join(gat_model_stats, 'train_losses' + str(model_GAT_choice)), 'a')
                    print(
                        'Training: loss = %.5f | Val: loss = %.5f' % (train_loss_avg / tr_size,
                                                                      val_loss_avg / vl_size))
                    print(
                        'Training: loss = %.5f | Val: loss = %.5f' % (train_loss_avg / tr_size,
                                                                      val_loss_avg / vl_size),
                        file=train_losses_file)

                    if epoch % CHECKPT_PERIOD == 0:
                        save_path = saver.save(sess, checkpt_file, global_step=epoch)
                        print("Training progress after %d epochs saved in path: %s" % (epoch, save_path))

            save_path = saver.save(sess, checkpt_file)
            print("Fully trained model saved in path: %s" % save_path)

            # restoring a pre-trained model
            saver.restore(sess, checkpt_file)
            print("Model restored.")

            ts_step = tr_size + vl_size
            ts_loss = 0.0

            while ts_step < ts_size + len(score_train) + len(score_val):
                (loss_value_ts,) = sess.run([loss],
                                            feed_dict={
                                                ftr_in: graphs_features[ts_step:ts_step + 1],
                                                bias_in: biases[ts_step:ts_step + 1],
                                                score_in: score_test[
                                                          ts_step - (vl_size + tr_size): ts_step + 1 - (
                                                                  vl_size + tr_size)],
                                                adj_in: adj_matrices[ts_step:ts_step + 1],
                                                is_train: False,
                                                attn_drop: 0.0,
                                                ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_step += 1

            print('Test loss:', ts_loss / ts_size)

            sess.close()


if __name__ == "__main__":
    hid_units = [10, 10, 15]
    n_heads = [2, 2, 3]
    model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                             n_heads=n_heads,
                                             nb_epochs=1500,
                                             edge_w_limit=200000)

    create_GAT_model(model_GAT_config)
