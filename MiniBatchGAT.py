from MainGAT import *
from GAT_hyperparam_config import GAT_hyperparam_config
CHECKPT_PERIOD = 25


def create_GAT_model(model_GAT_choice):
    # GAT model
    model = MainGAT()

    params = model_GAT_choice.params
    # data for adjancency matrices, node feature vectors, biases for masked attention and personality scores
    data, subjects = params['load_specific_data'](params)
    # nr of nodes for each graph: it is shared among all examples due to the dataset
    nb_nodes = data[subjects[0]]['adj'].shape[-1]
    # the initial length F of each node feature vector: for every graph, node feat.vecs. have the same length
    ft_size = data[subjects[0]]['feat'].shape[-1]
    # how many of the big-five personality traits the model is targeting at once
    outGAT_sz_target = len(params['pers_traits_selection'])

    subjects = shuffle_tr_data(subjects, len(subjects))
    # create a TensofFlow session, the context of evaluation for the Tensor objects
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))
            bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
            score_in = tf.placeholder(dtype=tf.float32, shape=(1, outGAT_sz_target))
            adj_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
            include_ew = tf.placeholder(dtype=tf.bool, shape=())

        prediction, unif_loss, excl_loss = model.inference_keras(in_feat_vects=ftr_in,
                                                           adj_mat=adj_in,
                                                           bias_mat=bias_in,
                                                           include_weights=include_ew,
                                                           hid_units=params['hidden_units'],
                                                           n_heads=params['attention_heads'],
                                                           target_score_type=outGAT_sz_target,
                                                           aggregator=params['readout_aggregator'],
                                                           residual=params['residual'],
                                                           activation=params['non_linearity'],
                                                           attn_drop=params['attn_drop'],
                                                           ffd_drop=params['ffd_drop'])

        loss = tf.losses.mean_squared_error(labels=score_in, predictions=prediction)
        # create tf session saver
        saver = tf.train.Saver()

        # minibatch operations
        zero_grads_ops, accum_ops, apply_ops = model.batch_training(loss=loss,
                                                                    u_loss=unif_loss,
                                                                    e_loss=excl_loss,
                                                                    lr=params['learning_rate'],
                                                                    l2_coef=params['l2_coefficient'])

        # number of training, validation, test graph examples
        split_sz = len(subjects) // 6
        tr_size, vl_size, ts_size = split_sz * 4, split_sz, split_sz
        print('The training size is: %d, the validation: %d and the test: %d' % (tr_size, vl_size, ts_size))

        # Create interactive session to execute the accumulation of gradients per batch
        sess = tf.InteractiveSession()
        # Necessary initializations
        tf.set_random_seed(1234)
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()).run()

        # record the minimum validation loss encountered until current epoch
        vlss_mn = np.inf
        # store the validation loss of previous epoch
        vlss_early_model = np.inf
        # record the number of consecutive epochs when the loss doesn't improve
        curr_step = 0

        # Train loop
        # nb_epochs - number of epochs for training: the number of iteration of gradient descent to optimize
        for epoch in range(1, params['num_epochs'] + 1):
            # number of iterations of the training set when batch-training
            tr_iterations = tr_size // params['batch_size']
            # Array for logging the training loss, the uniform loss, the exclusive loss
            tr_loss_log = np.zeros(tr_iterations)
            tr_uloss_log = np.zeros(tr_iterations)
            tr_eloss_log = np.zeros(tr_iterations)

            # shuffle the training dataset
            shuf_subjs = shuffle_tr_data(subjects, tr_size)

            for iteration in range(tr_iterations):
                params['attn_drop'] = 0.6
                params['ffd_drop'] = 0.6
                # Make sure gradients are set to 0 before entering minibatch loop
                sess.run(zero_grads_ops)
                # Loop over minibatches and execute accumulate-gradient operation
                for batch_step in range(params['batch_size']):
                    index = batch_step + iteration * params['batch_size']
                    sess.run([accum_ops], feed_dict={ftr_in: data[shuf_subjs[index]]['feat'],
                                                     bias_in: data[shuf_subjs[index]]['bias'],
                                                     score_in: data[shuf_subjs[index]]['score'],
                                                     adj_in: data[shuf_subjs[index]]['adj'],
                                                     include_ew: params['include_ew']})
                # Done looping over minibatches. Now apply gradients.
                sess.run(apply_ops)
                # Calculate the validation loss after every single batch training
                for batch_step in range(params['batch_size']):
                    index = batch_step + iteration * params['batch_size']
                    (tr_example_loss, u_loss, e_loss) = sess.run([loss, unif_loss, excl_loss],
                                                                 feed_dict={ftr_in: data[shuf_subjs[index]]['feat'],
                                                                            bias_in: data[shuf_subjs[index]]['bias'],
                                                                            score_in: data[shuf_subjs[index]]['score'],
                                                                            adj_in: data[shuf_subjs[index]]['adj'],
                                                                            include_ew: params['include_ew']})
                    tr_uloss_log[iteration] += u_loss
                    tr_eloss_log[iteration] += e_loss
                    tr_loss_log[iteration] += tr_example_loss

                tr_loss_log[iteration] /= params['batch_size']
                tr_eloss_log[iteration] /= params['batch_size']
                tr_uloss_log[iteration] /= params['batch_size']

            vl_avg_loss = 0.0
            params['attn_drop'] = 0.0
            params['ffd_drop'] = 0.0
            for vl_step in range(tr_size, tr_size + vl_size):
                (vl_example_loss,) = sess.run([loss], feed_dict={ftr_in: data[subjects[vl_step]]['feat'],
                                                                 bias_in: data[subjects[vl_step]]['bias'],
                                                                 score_in: data[subjects[vl_step]]['score'],
                                                                 adj_in: data[subjects[vl_step]]['adj'],
                                                                 include_ew: params['include_ew']})

                vl_avg_loss += vl_example_loss
            vl_avg_loss /= vl_size

            tr_avg_loss, tr_avg_uloss_log, tr_avg_eloss_log = map(lambda x: np.sum(x) / tr_iterations,
                                                                  [tr_loss_log, tr_uloss_log, tr_eloss_log])

            print('Training: loss = %.5f | Val: loss = %.5f | '
                  'Unifrom loss: %.5f| Exclusive loss: %.5f' % (
                      tr_avg_loss, vl_avg_loss, tr_avg_uloss_log, tr_avg_eloss_log))

            checkpt_file = model_GAT_choice.checkpt_file() + '_tester'
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
                if curr_step == params['patience']:
                    print('Early stop! Min loss: ', vlss_mn)
                    print('Early stop model validation loss: ', vlss_early_model)
                    break

    model_file = model_GAT_choice.checkpt_file() + '_tester_fullytrained'
    if not tf.train.checkpoint_exists(model_file):
        save_path = saver.save(sess, model_file)
        print("Fully trained model saved in path: %s" % save_path)

    # restoring a pre-trained model
    saver.restore(sess, model_file)
    print("Model restored.")

    ts_avg_loss = 0
    for ts_step in range(tr_size + vl_size, tr_size + vl_size + ts_size):
        (ts_example_loss,) = sess.run([loss], feed_dict={ftr_in: data[subjects[ts_step]]['feat'],
                                                         bias_in: data[subjects[ts_step]]['bias'],
                                                         score_in: data[subjects[ts_step]]['score'],
                                                         adj_in: data[subjects[ts_step]]['adj']})
        ts_avg_loss += ts_example_loss

    print('Test loss:', ts_avg_loss / ts_size)

    sess.close()


if __name__ == "__main__":
    hid_units = [20, 20, 10]
    n_heads = [3, 3, 2]
    aggregators = [MainGAT.concat_feature_aggregator]
    include_weights = [True]
    pers_traits = [['NEO.NEOFAC_A']]
    batches = [2]
    for aggr, iw, p_traits, batch_size in product(aggregators, include_weights, pers_traits, batches):
        dict_param = {
            'hidden_units': hid_units,
            'attention_heads': n_heads,
            'include_ew': iw,
            'readout_aggregator': aggr,
            'load_specific_data': load_struct_data,
            'pers_traits_selection': p_traits,
            'batch_size': batch_size,
            'edgeWeights_filter': None,
            'learning_rate': 0.0001,
        }

        create_GAT_model(GAT_hyperparam_config(dict_param))
