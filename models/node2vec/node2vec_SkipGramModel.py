import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pickle

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        #self.nodes_embed = nn.Embedding(emb_size, emb_dimension, sparse=True)
        #self.context_nodes_embed = nn.Embedding(emb_size, emb_dimension, sparse=True)

        #initrange = 1.0 / self.emb_dimension
        #init.uniform_(self.nodes_embed.weight.data, -initrange, initrange)
        #init.constant_(self.context_nodes_embed.weight.data, 0)

        initrange = 1.0 / self.emb_dimension
        nodes_init = torch.rand(self.emb_size, self.emb_dimension)
        context_nodes_init = torch.rand(self.emb_size, self.emb_dimension)
        init.uniform_(nodes_init, -initrange, initrange)
        init.constant_(context_nodes_init, 0)
        self.nodes_embed = nn.Parameter(nodes_init, requires_grad=True)
        self.context_nodes_embed = nn.Parameter(context_nodes_init, requires_grad=True)

    def forward1(self, pos_u, pos_v, neg_v):
        emb_u = self.nodes_embed(pos_u)
        emb_v = self.context_nodes_embed(pos_v)
        emb_neg_v = self.context_nodes_embed(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)



    def forward(self, source_node, target_node, label):
        label = label.to(self.device)

        source_embed = self.nodes_embed[source_node]
        target_embed = self.context_nodes_embed[target_node]

        inner_product = torch.sum(torch.mul(source_embed, target_embed), dim=1)
        pos_neg = torch.mul(label, inner_product)
        line_loss = F.logsigmoid(pos_neg)

        mean_loss = - torch.mean(line_loss)

        return mean_loss

    def save_embedding(self, id2word, file_name, emb_dict_file='emb_dict.pickle'):
        embedding = self.nodes_embed.cpu().data.numpy()

        embedding_map = {}
        for wid, w in id2word.items(): # todo warning: w in id2word is str(word)
            embedding_map[w] = embedding[wid]
        with open(emb_dict_file, 'wb') as file:
            pickle.dump(embedding_map, file)

        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
