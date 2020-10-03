

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve


def get_score(embs, node1, node2):
    #vector1 = embs[int(node1)]
    #vector2 = embs[int(node2)]
    try:
        vector1 = embs[node1]
        vector2 = embs[node2]
    except KeyError:
        vector1 = embs[str(node1)]
        vector2 = embs[str(node2)]
    return np.dot(vector1, vector2)  / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def evaluate(embs, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    for edge in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, edge[0], edge[1]))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    #print('Threshold: ', threshold)
    #print('len y_scores: ', len(y_scores))

    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps), fpr, tpr

