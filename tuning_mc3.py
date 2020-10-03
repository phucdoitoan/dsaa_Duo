
from tuner import custom_tuner, naive_tuner
#from trainer import Trainer

for train_percent in [3, 5, 7]:
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

        'iter_num': 20, #20, #50,

        'device': 'cuda:1',

        'dim': 64,
        #'order': order,
        'lr': 0.025,
        #'batch_number': 5000,
        'batch_size': 16, #32,
        'K': 5,
        'inner': 100,

        #'alpha': 0, #10,
        #'r': 1.5,

        'p': 1,
        'q': 1,

    }
    print(hyper)

    txt_file = path + 'model1=%s_model2=%s_best_config.txt' %(hyper['model1'], hyper['model2'])
    pkl_file = path + 'model1=%s_model2=%s_best_config.pkl' %(hyper['model1'], hyper['model2'])

    #custom_tuner(hyperpath, hyper, txt_file, pkl_file)
    naive_tuner(hyperpath, hyper, txt_file, pkl_file)