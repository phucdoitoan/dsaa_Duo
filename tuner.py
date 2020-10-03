

from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler

from trainer import Trainer

from time import time
import pickle

import sys, os
from contextlib import contextmanager
from itertools import product


def custom_tuner(hyperpath, hyper, txt_file, pkl_file):
	print(hyper)

	def sub_tuner(config):
		hyper['alpha'] = config['alpha']
		hyper['r'] = config['r']

		trainer = Trainer(hyperpath, hyper)

		for it in range(hyper['iter_num']):
			with suppress_stdout() as spr: #todo: dont need this one
				trainer.train_epoch()
			average_auc = (trainer.valid_auc1 + trainer.valid_auc2) / 2
			tune.track.log(mean_accuracy=average_auc)


	search_space = {
		'r': tune.grid_search([0.75, 1, 1.5, 3, 5]),
		'alpha': tune.grid_search([5, 10, 15, 20]),
	}

	analysis = tune.run(
		sub_tuner,
		config = search_space,
		resources_per_trial = {'gpu': 1}
		)

	best_config = analysis.get_best_config(metric='mean_accuracy')

	with open(txt_file, 'w') as file:
		file.write('\nr\talpha\n')
		file.write('%s\t%s' %(best_config['r'], best_config['alpha']))

	with open(pkl_file, 'wb') as file:
		pickle.dump(best_config, file)


@contextmanager
def suppress_stdout():
	with open(os.devnull, 'w') as devnull:
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:
			yield
		finally:
			sys.stdout = old_stdout



def naive_tuner(hyperpath, hyper, txt_file, pkl_file):

	r_space = [0.75, 1, 1.5, 3]
	alpha_space = [5, 10, 15, 20]

	best_r, best_alpha = None, None

	best_average_valid = 0

	print('Tuning ...')
	try:
		for r, alpha in product(r_space, alpha_space):
			#for alpha in alpha_space:
			hyper['r'] = r
			hyper['alpha'] = alpha

			trainer = Trainer(hyperpath, hyper) #todo: only need to re-initiate model wegihts -> save time of loading graphs in initial setups
			for it in range(hyper['iter_num']):
				trainer.train_epoch()

			average_valid = (trainer.valid_auc1 + trainer.valid_auc2) / 2

			if average_valid > best_average_valid:
				best_average_valid = average_valid
				best_r, best_alpha = r, alpha
	except Exception as e:
		print(e)
		print('Break at: r = %s, alpha = %s' %(r, alpha))
	finally:
		with open(txt_file, 'w') as file:
			file.write('\nr\talpha\tbest_average_valid\n')
			file.write('%s\t%s\t%.4f' %(best_r, best_alpha, best_average_valid))

		with open(pkl_file, 'wb') as file:
			pickle.dump({'r': best_r, 'alpha': best_alpha}, file)




