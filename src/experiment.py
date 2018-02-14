"""Tournament play experiment."""
from __future__ import absolute_import
import net_builder
import gp
import cPickle
import os
import numpy as np
# Use cuda ?
CUDA_ = True

if __name__=='__main__':
    # setup a tournament!
    nb_evolution_steps = 500
    tournament = \
        gp.TournamentOptimizer(
            population_sz=120,
            init_fn=net_builder.randomize_network,
            mutate_fn=net_builder.mutate_net,
            nb_workers=5,
            use_cuda=True)

    #USING LOGITS - CHANGE TO -1 FOR REGULAR OPERATION
    max_perf = 100
    arch_win = {}
    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        tournament.step()
        # keep track of the experiment results & corresponding architectures
        
        name = "tourney_{}".format(i)
        directory = "./tournament_results/testcompress2_24sensors/"
        if not os.path.exists(directory):
           os.makedirs(directory)
        cPickle.dump(tournament.stats, open(directory + name + '.stats','wb'))
        cPickle.dump(tournament.history, open(directory + name +'.pop','wb'))


    for i in xrange(len(tournament.stats)):
        #CHANGE TO > AND max FOR REGULAR OPERATION
        if min(tournament.stats[i])<max_perf:
           max_perf = min(tournament.stats[i])
           arch_win = tournament.history[i][np.argmin(tournament.stats[i])]

    print('\nMax performance was:{}'.format(max_perf))
    print('\nThe winning architecture was:{}'.format(arch_win))
