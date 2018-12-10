import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    plt.savefig('loss2.png')
    plt.show()

if __name__ == "__main__":
    plt_learn_proc()