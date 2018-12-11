import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ToolsStructural import get_filtered_struct_adjs
import seaborn as sns

gat_model_stats =  os.path.join(os.getcwd(), os.pardir, 'PartIIProject', 'stats')
if not os.path.exists(gat_model_stats):
    os.makedirs(gat_model_stats)

def plt_learn_proc():
    train_losses_file = os.path.join(gat_model_stats,'train_losses.txt')
    tr_loss, vl_loss = [],[]


    with open(train_losses_file,'r') as tr_loss_handle:
        for index,line in enumerate(tr_loss_handle,1):
            tr_loss.append(float(line.split()[3]))
            vl_loss.append(float(line.split()[8]))
    # Create data
    print(index)
    print(len(tr_loss))
    df = pd.DataFrame({'x': list(range(1, index+1)), 'y': np.array(tr_loss), 'y1':np.array(vl_loss)})

    # plot with matplotlib


    # Just load seaborn and the chart looks better:

    plt.plot('x', 'y', data=df, color='green', label = 'training loss')
    plt.plot('x', 'y1', data=df, color='red', label ='validation loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig('loss_woWeights.png')
    plt.show()

def edge_weight_distrib(limit):

    if os.path.exists('temp_file.npy'):
        edge_weights= np.load('temp_file.npy')
        print('load file')
    else:
        edge_weights = np.array(list(get_filtered_struct_adjs().values())).flatten()
        np.save('temp_file',edge_weights)

    edge_weights = list(map(int, edge_weights))


    filter_weights = [ew for ew in edge_weights if ew <limit]


    print('The ratio of elements under the limit %d is %f'%(limit,len(filter_weights)/len(edge_weights)))

if __name__ == "__main__":

    limits = list(range(0,2500000,10000))
    for limit in limits:
        pass
       # edge_weight_distrib(limit)

    plt_learn_proc()
