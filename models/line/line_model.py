

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class Line(nn.Module):

    #np.random.seed(42)

    def __init__(self, n, dim, order):
        super(Line, self).__init__()
        self.emb_size = n
        self.emb_dimension = dim
        self.order = order

        nodes_init = np.random.uniform(-1, 1, (n, dim)).astype(np.float32)
        self.nodes_embed = nn.Parameter(torch.from_numpy(nodes_init), requires_grad=True)

        if self.order == 2:
            context_init = np.random.uniform(-1, 1, (n, dim)).astype(np.float32)
            self.context_nodes_embed = nn.Parameter(torch.from_numpy(context_init), requires_grad=True)


    def forward(self, source_node, target_node, label):
        """

        :param source_node: list of [i,i,i,i,i, ...] of source nodes: each source node repeat K + 1 time: one for target node, K times for K negative nodes
        :param target_node: list of [j,j1,j2,..,jK, ...] of target nodes: j is target node, j1 -> jK is negative nodes
        :param label: FloatTensor([1, -1, -1, -1, -1, -1, 1, ....]) label to indicate which is target node, which is negative nodes
        :return:
        """
        label = label.to(self.device)

        source_embed = self.nodes_embed[source_node]

        if self.order == 1:
            target_embed = self.nodes_embed[target_node]

        elif self.order == 2:  # self.order == 2
            target_embed = self.context_nodes_embed[target_node]
        else:
            print("ERROR: order has to be 1 or 2")

        inner_product = torch.sum(torch.mul(source_embed, target_embed), dim=1)
        pos_neg = torch.mul(label, inner_product)
        line_loss = F.logsigmoid(pos_neg)

        mean_loss = - torch.mean(line_loss)

        return mean_loss


    def save_embed(self, embedding_mapping, embed_file):

        embedding = self.nodes_embed.data
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        normalized_embedding = normalized_embedding.to('cpu')
        embed_dict = embedding_mapping(normalized_embedding.numpy())
        #todo: why pickle dump eat up Memory with torch.tensor but not with numpy?? (i.e. remove numpy() in the previous line)
        pickle.dump(embed_dict, open(embed_file, 'wb'))

