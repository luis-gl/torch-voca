import pickle
import matplotlib.pyplot as plt

def plot_loss(model_dic1, model_dic2, title=None, save=False, test=False):
    train_loss_history1 = model_dic1['train_loss_history']
    train_loss_history2 = model_dic2['train_loss_history']
    n1 = len(train_loss_history1)
    n2 = len(train_loss_history2)
    n = n1 if n1 < n2 else n2
    x_values = range(1, n + 1)
    plt.figure(figsize=(7, 5))
    if title is None:
        plt.title('model 1 vs model 2 Loss')
    else:
        plt.title(title + ' Loss')
    plt.plot(x_values, train_loss_history1[:n], '-o', label='special')
    plt.plot(x_values, train_loss_history2[:n], '-o', label='common')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()
    if save:
        if test:
            fname = './plots/{}_losses_test.png'.format(title)
        else:
            fname = './plots/{}_losses.png'.format(title)
        plt.savefig(fname)

def main():
    with open('model_losses.pickle', 'rb') as handle:
        model_dict1 = pickle.load(handle)

    with open('model_losses_common_bn.pickle', 'rb') as handle2:
        model_dict2 = pickle.load(handle2)
    
    plot_loss(model_dict1, model_dict2, 'VOCA special vs common batchnorm in 10 epochs', save=True)

if __name__ == '__main__':
    main()