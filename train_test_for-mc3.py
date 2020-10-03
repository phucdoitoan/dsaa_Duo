
from trainer import Trainer
#from trainer import Trainer
import torch
import pickle

for train_percent in [7, 5, 3]:
    path = 'data/mc3/train=0.%s/' %(train_percent)

    hyperpath = {
        'full_file1': path + 'data/calls_500.pkl',
        'train_file1': path + 'data/calls_500_%s0remained_weighted.pkl' % train_percent,
        'valid_file1': path + 'data/calls_500_%s0removed_valid.pkl' % train_percent,
        'test_file1': path + 'data/calls_500_%s0removed_test.pkl' % train_percent,

        'full_file2': path + 'data/emails_500.pkl',
        'train_file2': path + 'data/emails_500_%s0remained_weighted.pkl' % train_percent,
        'valid_file2': path + 'data/emails_500_%s0removed_valid.pkl' % train_percent,
        'test_file2': path + 'data/emails_500_%s0removed_test.pkl' % train_percent,

        'embed_file1': path + 'embedding/embedding_order=%s_%s0_G1.pkl' % (' ', train_percent),
        'embed_file2': path + 'embedding/embedding_order=%s_%s0_G2.pkl' % (' ', train_percent),
    }

    hyper = {
        'model1': 'node2vec',
        'model2': 'node2vec',
        #'model1': 'line-2',
        #'model2': 'line-2',

        'iter_num': 20, #50,

        'dim': 64,
        'lr': 0.025,
        'batch_size': 16, #32,
        'K': 5,
        'inner': 100,

        'alpha': 5, #20, #0,
        'r': 1.5,

        'p': 1,
        'q': 1,

    }
    best_config = False
    zero_r = False #True

    if best_config:
        print('**** TEST WITH BEST CONFIG *****')
        config_file = path + 'model1=%s_model2=%s_best_config.pkl' %(hyper['model1'], hyper['model2'])
        best_config = pickle.load(open(config_file, 'rb'))
        hyper['r'] = best_config['r']
        hyper['alpha'] = best_config['alpha']
        if zero_r:
            hyper['r'] = 0
    else:
        print('TEST WITH MANUALLY CONFIGED alpha = %s' %hyper['alpha'])

    print(hyper)

    auc1 = []
    auc2 = []
    repeat = 5
    for _ in range(repeat):
        print('train percent: ', train_percent)
        trainer = Trainer(hyperpath, hyper)
        for it in range(hyper['iter_num']):
            print('\n********** Iter %s: ***********' %(it))
            trainer.train_epoch()
        #trainer.normalize_and_save()
        trainer.test()

        auc1.append(trainer.test_auc1.item())
        auc2.append(trainer.test_auc2.item())

    mean_auc1 = torch.mean(torch.tensor(auc1))
    std_auc1 = torch.std(torch.tensor(auc1))

    mean_auc2 = torch.mean(torch.tensor(auc2))
    std_auc2 = torch.std(torch.tensor(auc2))

    auc_file = path + 'auc/auc_model=%s-%s_r=%s_alpha=%s.txt' %(hyper['model1'], hyper['model2'], hyper['r'], hyper['alpha'])

    with open(auc_file, 'w') as file:
        file.write('\nauc1\tstd1\tauc2\tstd2\n')
        file.write('%.4f\t%.4f\t%.4f\t%.4f' %(mean_auc1, std_auc1, mean_auc2, std_auc2))



