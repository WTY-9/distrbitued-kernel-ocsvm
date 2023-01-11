import numpy as np
from sklearn.metrics import (roc_auc_score, precision_score,
    average_precision_score, recall_score, f1_score, confusion_matrix)
'''
横轴预测结果，纵轴实际结果
     0     1
0    TP    FN
1    FP    TN
precison = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = (2 * precision * recall)/(precision + recall)

TPR=recall
FPR=FP/(FP+TN)

'''
def metrics(y_test, out_y, y_scores):
    result = dict()
    result['Confusion matrix'] = confusion_matrix(-y_test, -out_y, labels=[1, -1])
    TP = result['Confusion matrix'][0, 0]
    FN = result['Confusion matrix'][0, 1] 
    FP = result['Confusion matrix'][1, 0]
    TN = result['Confusion matrix'][1, 1]
    
    result['TPR'] = TP / (TP + FN)
    result['FPR'] = FP / (TN + FP)
    result['F1'] = f1_score(-y_test, -out_y)
    result['AUROC'] = roc_auc_score(-y_test, -y_scores)
    result['Precision'] = precision_score(-y_test, -out_y)
    result['Recall'] = recall_score(-y_test, -out_y)
    result['AUPRC'] = average_precision_score(-y_test, -y_scores)

    return result