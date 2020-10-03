
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import custome_Graph
import utils
from node2vec_SkipGramModel import SkipGramModel
import networkx as nx
from time import time

def read_nxgraph(input_file):
    G = nx.read_gpickle(input_file)
    if not nx.is_weighted(G, weight='weight'):
        print('G is not weighted -> assign weight 1 to each edge')
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    else:
        print('G is weighted')
    return G


def main():
    #input_file = 'data/edges3437_50remained_G1.pkl'
    #input_file = 'data/facebook_4k_50remained_G1.pkl'
    #input_file = 'data/1k_fb_50remained.pkl'
    input_file = 'data/calls_500_50remained.pkl'
    #input_file = 'data/emails_500_50remained.pkl'
    nx_G = read_nxgraph(input_file)
    G = custome_Graph.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    t0 = time()
    with open('tmp.txt', 'w') as file:
        for walk in walks:
            for i in walk:
                file.write('%s ' % (i))
            file.write('\n')
    print('write into tmp.txt file in %.4fs' % (time() - t0))
    data = utils.DataReader('tmp.txt', min_count=0)
    dataset = utils.Word2vecDataset(data, window_size=5)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            num_workers=0, collate_fn=dataset.collate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    emb_size = len(data.word2id)
    emb_dimension = 64 #128
    skip_gram_model = SkipGramModel(emb_size, emb_dimension).to(device)
    skip_gram_model.device = device

    for iteration in range(16):

        print("\n\n\nIteration: " + str(iteration + 1))
        #optimizer = optim.SparseAdam(skip_gram_model.parameters(), lr=0.001)
        optimizer = optim.Adam(skip_gram_model.parameters(), lr=0.025)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

        running_loss = 0.0
        for i, sample_batched in enumerate(tqdm(dataloader)):
            if len(sample_batched[0]) > 1:
                pos_u = sample_batched[0]
                pos_v = sample_batched[1]
                neg_v = sample_batched[2]

                scheduler.step()
                optimizer.zero_grad()
                loss = skip_gram_model.forward(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()

                running_loss = running_loss * 0.9 + loss.item() * 0.1
                if i > 0 and i % 500 == 0:
                    print(" Loss: " + str(running_loss))

    skip_gram_model.save_embedding(data.id2word, 'out.vec')


if __name__ == '__main__':
    main()