
import networkx as nx
import pickle
from evaluate import evaluate
import time
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

def numerical_integral(l_x, l_y):
    integral = 0
    for i in range(len(l_x)-1):
        sum_i = (l_x[i+1]-l_x[i])*(l_y[i+1]+l_y[i]) / 2
        integral += sum_i

    return integral

def evaluate_main(embeddings=None, embed_file= 'data/embedding=tensorflow_fb_remained_order-1.pkl', G_full_file='data/facebook_combined.pkl', G_removed_file='data/facebook_removed.pkl'):
    """

    :param embeddings:  need to be dictionary of embeddings
    :param embed_file:
    :param G_full_file:
    :param G_removed_file:
    :return:
    """
    G_full = nx.read_gpickle(G_full_file)
    #G_remained = nx.read_gpickle(G_remained_file)
    G_removed = nx.read_gpickle(G_removed_file)

    #print('Loaded graphs')

    if embeddings == None:
        with open(embed_file, 'rb') as file:
            embeds = pickle.load(file)
    else:
        #print('\t\t\t\t\t\t\t ***** Using embed_dict to predict link *********')
        embeds = embeddings
    #print('Loaded embeds')

    #t1_neg = timeit.default_timer()
    negative_set = set()
    #for i in tqdm(range(len(G_removed.edges))):
    for i in range(len(G_removed.edges)):                                                 # THIS IS THE PART WHICH COST THE MOST TIME OF THE WHOLE PROGRAM -> PARALLEL
    #while len(negative_set) < len(G_removed.edges):
        #print(len(negative_set))
        #while True:
        while len(negative_set) < (i+1):
            x = random.sample(list(G_full.nodes),1)[0]
            y = random.sample(list(G_full.nodes),1)[0]
            #print('negative edges (%d, %d)' %(x, y))
            if (x, y) not in G_full.edges:
                if G_full.__class__ == nx.Graph:
                    if (y, x) not in negative_set:
                        negative_set.add((x, y))
                else:
                    negative_set.add((x, y))

    #t2_neg = timeit.default_timer()
    #print('Done randomly choosing negative edges: %.2f s' %(t2_neg - t1_neg))

    positive_set = set(G_removed.edges)

    #intersec = positive_set.intersection(negative_set)
    #uni = positive_set.union(negative_set)
    #print('len postive and negative intersec ', len(intersec))
    #print('len pos and neg union ', len(uni))

    false_edges = list(negative_set)
    true_edges = list(G_removed.edges)

    #print('len false edges: ', len(false_edges))
    #print('len true edges: ', len(true_edges))

    #t1 = timeit.default_timer()
    auc_score, f1_score, auc, fpr, tpr = evaluate(embeds, true_edges, false_edges)
    #t2 = timeit.default_timer()
    #print('Evaluate in %.2f s' %(t2-t1))

    return auc_score, f1_score, auc


def average_evaluate(embed_file, full_file, removed_file, embeddings=None, repeat=100):
    list_auc = []
    for _ in range(repeat):
        auc, _, _ = evaluate_main(embeddings=embeddings, embed_file=embed_file, G_full_file=full_file, G_removed_file=removed_file)
        list_auc.append(auc)

    list_auc = np.asarray(list_auc)
    auc_mean, auc_std = np.mean(list_auc), np.std(list_auc)

    if __name__ == '__main__':
        print(' mean of auc: ', auc_mean)
        print(' std of auc: ', auc_std)

    return auc_mean, auc_std


if __name__ == '__main__':
    order = 'gwl'
    epoch = 149
    prefix = 'G1'
    AUC_file = 'data/small_fb/gwl/gwl_AUC_%s_dim=20_epoch=%s.png' %(prefix, epoch)
    embed_file = 'data/small_fb/gwl/gwl_embeddings_edges3437_%s_dim=20_epoch=%s.pkl' %(prefix, epoch)
    # embed_file= 'data/small_fb/gwl/gwl_embeddings_edges3437_G2.pkl'
    G_full_file = 'data/small_fb/gwl/edges3437_full.pkl'
    # G_remained_file = 'data/small_fb/edges348_remained_%10_1.pkl'
    G_removed_file = 'data/small_fb/gwl/edges3437_50removed_test_%s.pkl' %prefix

    #evaluate_main(embed_file, G_full_file, G_removed_file)
    average_evaluate(embed_file, G_full_file, G_removed_file)






