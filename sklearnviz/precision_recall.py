from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def plot_precision_recall(y, y_pred, spacing=0.2):
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    roc_auc = auc(recall, precision)

    plt.figure(figsize=(10,10))
    plt.title('Precision vs Recall Curve', fontsize=18)
    plt.plot(recall, precision, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Recall', fontsize=16)
    
    acc = 0
    euc = spacing
    lx = 0 
    ly = 0
    for idx, t in enumerate(thresholds):
        if acc >= spacing or idx == len(thresholds)-1:
            plt.text(recall[idx], 
                     precision[idx], 
                     '%0.2f' % t, 
                     backgroundcolor='lightgray', 
                     color='black')
            acc = 0
        else:
            acc += euc
            
        euc = ((recall[idx] - lx)**2 + (precision[idx] - ly)**2)**0.5
        lx = recall[idx]
        ly = precision[idx]

    plt.show()