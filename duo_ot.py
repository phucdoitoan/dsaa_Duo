

import torch
import torch.nn as nn


class Duo_OT(nn.Module):

    def __init__(self, model1, model2):

        super(Duo_OT, self).__init__()

        self.model1 = model1
        self.model2 = model2

    def sinkhorn_loss(self, batch1, batch2, P, has_context=True):

        source1, target1, _ = batch1
        source2, target2, _ = batch2

        if not has_context:
            nodes1 = list(set(source1).union(set(target1)))
            nodes2 = list(set(source2).union(set(target2)))
            context_nodes1 = []
            context_nodes2 = []
        else:
            #print('\t\t\t ORDER 2')
            nodes1 = list(set(source1))
            nodes2 = list(set(source2))
            context_nodes1 = list(set(target1))
            context_nodes2 = list(set(target2))


        # OT loss for node embedding M(X1, X2, P)
        emb1 = self.model1.nodes_embed
        emb2 = self.model2.nodes_embed


        X1_row = emb1[nodes1].unsqueeze(1)
        X2_col_detach = emb2.detach().unsqueeze(0)
        dist_1 = torch.sum(torch.abs(X1_row - X2_col_detach)**2, dim=2)
        ot_cost_1 = torch.sum(torch.mul(dist_1, P[nodes1]), dim=1)

        X2_col = emb2[nodes2].unsqueeze(0)
        X1_row_detach = emb1.detach().unsqueeze(1)
        dist_2 = torch.sum(torch.abs(X2_col - X1_row_detach)**2, dim=2)
        ot_cost_2 = torch.sum(torch.mul(dist_2, P[:, nodes2]), dim=0)

        M = torch.mean(ot_cost_1) + torch.mean(ot_cost_2)

        loss = M

        # OT loss for context node embedding M(X'1, X'2, P)
        if has_context:  # if there is context embeddings
            #print('\t\t\t ORDER 2')
            context_emb1 = self.model1.context_nodes_embed
            context_emb2 = self.model2.context_nodes_embed

            context_X1_row = context_emb1[context_nodes1].unsqueeze(1)
            context_X2_col_detach = context_emb2.detach().unsqueeze(0)
            context_dist_1 = torch.sum(torch.abs(context_X1_row - context_X2_col_detach) ** 2, dim=2)
            context_ot_cost_1 = torch.sum(torch.mul(context_dist_1, P[context_nodes1]), dim=1)

            context_X2_col = context_emb2[context_nodes2].unsqueeze(0)
            context_X1_row_detach = context_emb1.detach().unsqueeze(1)
            context_dist_2 = torch.sum(torch.abs(context_X2_col - context_X1_row_detach) ** 2, dim=2)
            context_ot_cost_2 = torch.sum(torch.mul(context_dist_2, P[:, context_nodes2]), dim=0)

            context_M = torch.mean(context_ot_cost_1) + torch.mean(context_ot_cost_2)

            loss += context_M


        return loss



    def forward(self, batch1, batch2, has_context, P, alpha):

        loss1 = self.model1(*batch1)  # need to unroll batch to fit model1's, model2's inputs
        loss2 = self.model2(*batch2)


        if alpha == 0.:
            loss = loss1 + loss2
        else:
            loss_M = self.sinkhorn_loss(batch1, batch2, P, has_context)  # can be customed with other Optimal Tranpsort approximation method

            loss = loss1 + loss2 + alpha*loss_M

        return loss

