from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_s(c):
    num = c['rate_negative'] * (c['tn_util'] - c['fp_util'])
    den = c['rate_positive'] * (c['tp_util'] - c['fn_util'])
    return num / den

def plot_roc(y, y_pred, spacing=0.2, indifference=None, d=1):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    
    acc = 0
    euc = spacing
    lx = 0 
    ly = 0
    for idx, t in enumerate(thresholds):
        if acc >= spacing or idx == len(thresholds)-1:
            plt.text(false_positive_rate[idx], 
                     true_positive_rate[idx], 
                     '%0.2f' % t, 
                     backgroundcolor='lightgray', 
                     color='black')
            acc = 0
        else:
            acc += euc
            
        euc = ((false_positive_rate[idx] - lx)**2 + (true_positive_rate[idx] - ly)**2)**0.5
        lx = false_positive_rate[idx]
        ly = true_positive_rate[idx]
    
    if indifference:
        slope = get_s(indifference)
        xs = [0.01 * x for x in range(101)]
        i = d - slope
        ys = [(slope * x) + i for x in xs]
        plt.plot(xs, ys, label='Indifference Curve')
        plt.legend()
    
    plt.show()