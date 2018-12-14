import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ToolsStructural import get_filtered_struct_adjs
from SupervisedGAT import GAT_hyperparam_config, gat_model_stats


def plt_learn_proc(model_GAT_config):
    train_losses_file = os.path.join(gat_model_stats, 'train_losses' + str(model_GAT_config))
    tr_loss, vl_loss = [], []

    with open(train_losses_file, 'r') as tr_loss_handle:
        for index, line in enumerate(tr_loss_handle, 1):
            tr_loss.append(float(line.split()[3]))
            vl_loss.append(float(line.split()[8]))
    # Create data
    print(index)
    print(len(tr_loss))
    df = pd.DataFrame({'x': list(range(1, index + 1)), 'y': np.array(tr_loss), 'y1': np.array(vl_loss)})

    # plot with matplotlib

    # Just load seaborn and the chart looks better:

    plt.plot('x', 'y', data=df, color='green', label='training loss')
    plt.plot('x', 'y1', data=df, color='red', label='validation loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig('loss_woWeights.png')
    plt.show()


def edge_weight_distrib(limit):
    if os.path.exists('temp_file.npy'):
        edge_weights = np.load('temp_file.npy')
        print('load file')
    else:
        edge_weights = np.array(list(get_filtered_struct_adjs().values())).flatten()
        np.save('temp_file', edge_weights)

    edge_weights = list(map(int, edge_weights))

    filter_weights = [ew for ew in edge_weights if ew < limit]

    print('The ratio of elements under the limit %d is %f' % (limit, len(filter_weights) / len(edge_weights)))


if __name__ == "__main__":
    hid_units = [64, 32, 16]
    n_heads = [4, 4, 6]
    model_GAT_config = GAT_hyperparam_config(hid_units=hid_units,
                                             n_heads=n_heads,
                                             dataset_type='structural',
                                             nb_epochs=1500,
                                             edge_w_limit=80000)

    plt_learn_proc(model_GAT_config)
