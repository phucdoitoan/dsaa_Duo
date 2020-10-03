
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import pickle
import networkx as nx

from sinkhorn.sinkhorn import SinkhornDistance

from models.line.line_model import Line
from models.line.utils import LineDataset

from models.node2vec import utils as n2v_utils
from models.node2vec import custome_Graph
from models.node2vec.node2vec_SkipGramModel import SkipGramModel

from duo_ot import Duo_OT
from test_evaluate import average_evaluate

from time import time
from tqdm import tqdm

class Trainer():

    def __init__(self, hyperpath, hyper):
        self.full_file1 = hyperpath['full_file1']
        self.train_file1 = hyperpath['train_file1']
        self.valid_file1 = hyperpath['valid_file1']
        self.test_file1 = hyperpath['test_file1']

        self.full_file2 = hyperpath['full_file2']
        self.train_file2 = hyperpath['train_file2']
        self.valid_file2 = hyperpath['valid_file2']
        self.test_file2 = hyperpath['test_file2']

        self.embed_file1 = hyperpath['embed_file1']
        self.embed_file2 = hyperpath['embed_file2']

        self.dim = hyper['dim']  # 64
        #self.order = hyper['order']  # 2
        self.learning_rate = hyper['lr']  # 0.025
        #self.batch_number = hyper['batch_number']  # 1500

        self.batch_size = hyper['batch_size']  # 32
        self.K = hyper['K']  # 5

        self.inner = hyper['inner']  # 100
        self.iter_num = hyper['iter_num']

        # need-to-tune hyperparameters
        self.alpha = hyper['alpha']  # 5
        self.r = hyper['r']  # 1.5

        self.p, self.q = hyper['p'], hyper['q']

        try:
            self.device = hyper['device']
        except:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dataset1, self.model1, has_context1 = self.set_model(hyperpath['train_file1'], hyper['model1'], model_no=1)
        self.dataset2, self.model2, has_context2 = self.set_model(hyperpath['train_file2'], hyper['model2'], model_no=2)
        # todo: need to fix model_no arg, either creating optim and scheduler inside set_model or find a way to remove model_no arg

        self.has_context = (has_context1 and has_context2)
        print('self.has_context: ', self.has_context)

        self.batch_size1 = self.batch_size
        self.batch_size2 = int(self.batch_size1 * len(self.dataset2)/len(self.dataset1))

        self.dataloader1 = DataLoader(self.dataset1, batch_size=self.batch_size1, shuffle=True, collate_fn=self.dataset1.collate)
        self.dataloader2 = DataLoader(self.dataset2, batch_size=self.batch_size2, shuffle=True, collate_fn=self.dataset2.collate)


        print('len dataset1: ', len(self.dataset1))
        print('len dataset2: ', len(self.dataset2))
        print('len dataloader1: ', len(self.dataloader1))
        print('len dataloader2: ', len(self.dataloader2))
        print('batch_size 1: %s' %(self.batch_size1))
        print('batch_size 2: %s' %(self.batch_size2))



        #self.n1, self.n2 = self.dataset1.num_of_nodes, self.dataset2.num_of_nodes

        self.batch_number1 = self.iter_num * len(self.dataloader1)
        self.batch_number2 = self.iter_num * len(self.dataloader2)

        print('batch_number: ', self.batch_number1)
        print('batch_number2: ', self.batch_number2)

        self.duo_model = Duo_OT(self.model1, self.model2)

        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=self.learning_rate)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=self.learning_rate)

        self.scheduler1 = lr_scheduler.LambdaLR(self.optimizer1,
                                           (lambda b: 1 - (b-1) / self.batch_number1 if 1 - (b-1) / self.batch_number1 > 0.0001 else 0.0001))
        self.scheduler2 = lr_scheduler.LambdaLR(self.optimizer2,
                                           (lambda b: 1 - (b-1) / self.batch_number2 if 1 - (b-1) / self.batch_number2 > 0.0001 else 0.0001))
        # todo: how to change optimizer and scheduler for node2vec

        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=50, device=self.device)
        self.mu1, self.mu2 = torch.from_numpy(self.dataset1.node_distribution_power(self.r)), \
                             torch.from_numpy(self.dataset2.node_distribution_power(self.r))
        # todo: how to get m1, mu2 for node2vec

        self.P = torch.zeros(self.model1.emb_size, self.model2.emb_size).to(self.device)

        self.loss_list = []
        self.skn_list = []

        self.valid_auc1, self.valid_auc2 = 0, 0


    def set_model(self, train_file, model_type, model_no):  
        if model_type == 'line-1':
            has_context = False
            dataset = LineDataset(train_file)
            model = Line(dataset.num_of_nodes, self.dim, order=1)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            #scheduler =
        elif model_type == 'line-2':
            has_context = True
            dataset = LineDataset(train_file)
            model = Line(dataset.num_of_nodes, self.dim, order=2)
        elif model_type == 'node2vec':
            has_context = True
            nx_G = nx.read_gpickle(train_file)
            if not nx.is_weighted(nx_G, weight='weight'):  # todo: find another way to check if G is weighted
                print('G is not weighted -> assign weight 1 to each edge')
                for edge in nx_G.edges():
                    nx_G[edge[0]][edge[1]]['weight'] = 1
            else:
                print('G is weighted')

            G = custome_Graph.Graph(nx_G, is_directed=False, p=self.p, q=self.q)
            G.preprocess_transition_probs()
            walks = G.simulate_walks(num_walks=10, walk_length=80) #todo: original 10, 80
            #t0 = time()
            tmp_file = 'tmp-%s.txt' %model_no # todo: same name for tmp.txt -> overwrite by the second model
            with open(tmp_file, 'w') as file:   
                for walk in walks:
                    for i in walk:
                        file.write('%s ' % (i))
                    file.write('\n')
            #print('write into tmp.txt file in %.4fs' % (time() - t0))
            data = n2v_utils.DataReader(tmp_file, min_count=0)
            dataset = n2v_utils.Word2vecDataset(data, window_size=self.K)

            model = SkipGramModel(len(data.word2id), self.dim)
        else:
            raise NotImplementedError("Sub-model '%s' is not implemented! Implemented models: %s, %s, %s" %(model_type, 'line-1', 'line-2', 'node2vec'))

        model.device = self.device #add device attribute to model

        return dataset, model.to(self.device), has_context


    def train_epoch(self):
        #for it in range(self.iter_num):
        t0 = time()
        total_loss = 0
        size = min(len(self.dataloader1), len(self.dataloader2))
        for i, (batch1, batch2) in enumerate(zip(self.dataloader1, self.dataloader2)):

            self.optimizer1.zero_grad(), self.optimizer2.zero_grad()
            loss = self.duo_model(batch1=batch1, batch2=batch2,
                                  has_context=self.has_context, P=self.P, alpha=self.alpha)
            loss.backward()
            self.optimizer1.step(), self.optimizer2.step()
            self.scheduler1.step(), self.scheduler2.step()

            if (i % self.inner == (self.inner-1)) or (i == size-1): # update sinkhorn and update_best embedding after innder batches or when finishing one epoch
                if self.alpha != 0:
                    self.train_sinkhorn()
                self.update_best()

            #print('\t\t\t i: %s, loss: %s' %(i, loss.item()))

            total_loss += loss.detach().item()

        print('\t Loss: %.4f in %4.f s - min len dataloader: %s' %(total_loss/min(len(self.dataloader1), len(self.dataloader2)), time() - t0, size))
        #print('\t in %4.f s - min len dataloader: %s' %(time() - t0, size))


    def train_sinkhorn(self):
        with torch.no_grad():
            if not self.has_context:
                X1 = self.model1.nodes_embed
                X2 = self.model2.nodes_embed
            else:
                X1 = torch.cat((self.model1.nodes_embed, self.model1.context_nodes_embed), dim=-1)
                X2 = torch.cat((self.model2.nodes_embed, self.model2.context_nodes_embed), dim=-1)

            skn_dist, self.P, _ = self.sinkhorn(X1, X2, self.mu1, self.mu2)
            #self.skn_list.append(skn_dist.to('cpu').item())
            #remove comment to save sinkhorn distance


    def update_best(self, out_it=0):
        with torch.no_grad():
            embed_dict1 = self.dataset1.embedding_mapping(self.model1.nodes_embed.data.clone().to('cpu').numpy())
            embed_dict2 = self.dataset2.embedding_mapping(self.model2.nodes_embed.data.clone().to('cpu').numpy())
            # todo: if do not use clone(), embed_dict will update along with value of nodes_embed

            auc1, std1 = average_evaluate('no need for file', full_file=self.full_file1,
                                          removed_file=self.valid_file1,
                                          embeddings=embed_dict1, repeat=3)
            auc2, std2 = average_evaluate('no need for file', full_file=self.full_file2,
                                          removed_file=self.valid_file2,
                                          embeddings=embed_dict2, repeat=3)

            # todo : without .clone(), best_embed updates along with nodes_embed
            # todo: abundant to save both embed_dict and best_embed_dict
            if auc1 > self.valid_auc1:
                self.valid_auc1 = auc1
                #self.valid_std1 = std1
                self.best_embed1 = self.model1.nodes_embed.data.clone()
                self.best_embed_dict1 = embed_dict1
                #self.best_outer1 = out_it
                self.best_P1 = self.P
                print('   max valid auc1: %.4f, std: %.4f' %(auc1, std1))

            if auc2 > self.valid_auc2:
                self.valid_auc2 = auc2
                #self.valid_std2 = std2
                self.best_embed2 = self.model2.nodes_embed.data.clone()
                self.best_embed_dict2 = embed_dict2
                #self.best_outer2 = out_it
                self.best_P2 = self.P
                print('   max valid auc2: %.4f, std: %.4f' %(auc2, std2))


    def normalize_and_save(self):
        normalized1 = F.normalize(self.best_embed1, p=2, dim=1).to('cpu').numpy()
        normalized2 = F.normalize(self.best_embed2, p=2, dim=1).to('cpu').numpy()

        embed_dict1 = self.dataset1.embedding_mapping(normalized1)
        embed_dict2 = self.dataset2.embedding_mapping(normalized2)
        
        pickle.dump(embed_dict1, open(self.embed_file1, 'wb'))
        pickle.dump(embed_dict2, open(self.embed_file2, 'wb'))


    def test(self):
        print('best valid auc1: ', self.valid_auc1)
        print('best valid auc2: ', self.valid_auc2)

        auc1, std1 = average_evaluate(embed_file='', full_file=self.full_file1,
                                      removed_file=self.test_file1, embeddings=self.best_embed_dict1, repeat=10)
        auc2, std2 = average_evaluate(embed_file='', full_file=self.full_file2,
                                      removed_file=self.test_file2, embeddings=self.best_embed_dict2, repeat=10)

        self.test_auc1, self.test_std1 = auc1, std1
        self.test_auc2, self.test_std2 = auc2, std2

        print('        test auc1: %.4f, std: %.4f ' % (auc1, std1))
        print('        test auc2: %.4f, std: %.4f ' % (auc2, std2))


    def save_results(self, auc_file, loss_file):
        best_auc = {
            'alpha': self.alpha,
            'r': self.r,

            'valid_auc1': self.valid_auc1,
            #'valid_std1': self.valid_std1,
            #'max_outer1': self.max_batch1,
            'test_auc1': self.test_auc1,
            'test_std1': self.test_std1,

            'valid_auc2': self.valid_auc2,
            #'valid_std2': self.valid_std2,
            #'max_outer2': self.max_batch2,
            'test_auc2': self.test_auc2,
            'test_std2': self.test_std2,
        }

        best_loss_P = {
            'best_P1': self.best_P1.to('cpu'),
            'best_P2': self.best_P2.to('cpu'),
            'loss_list': self.loss_list,
            'skn_list': self.skn_list,
        }

        with open(auc_file, 'wb') as file:
            pickle.dump(best_auc, file)

        with open(loss_file, 'wb') as file:
            pickle.dump(best_loss_P, file)


    """
    def train_inner(self, inner):
        for in_it in range(inner):
            source1, target1, label1 = self.dataloader1.fetch_batch(batch_size=self.batch_size1, K=self.K)
            source2, target2, label2 = self.dataloader2.fetch_batch(batch_size=self.batch_size2, K=self.K)

            label1, label2 = torch.FloatTensor(label1).to(self.device), \
                             torch.FloatTensor(label2).to(self.device)

            self.optimizer1.zero_grad(), self.optimizer2.zero_grad()

            loss = self.duo_model(batch1=(source1, target1, label1),
                                  batch2=(source2, target2, label2),
                                  has_context=(self.order==2), P=self.P, alpha=self.alpha)
                                  # todo: define self.has_context acc to self.order or self.model
            loss.backward()

            self.optimizer1.step(), self.optimizer2.step()
            self.scheduler1.step(), self.scheduler2.step()

            if in_it == 0:
                print('\t Loss: %s' % (loss.item()))
                self.loss_list.append(loss.item())
    """