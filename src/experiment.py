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
    nb_evolution_steps = 300
    tournament = \
        gp.TournamentOptimizer(
            population_sz=250,
            init_fn=net_builder.randomize_network,
            mutate_fn=net_builder.mutate_net,
            nb_workers=20,
            use_cuda=True)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        tournament.step()
        # keep track of the experiment results & corresponding architectures
        
        name = "tourney_{}".format(i)
        directory = "./tournament_results/pop250/"
        if not os.path.exists(directory):
           os.makedirs(directory)
        cPickle.dump(tournament.stats, open(directory + name + '.stats','wb'))
        cPickle.dump(tournament.history, open(directory + name +'.pop','wb'))
        
    print('\nMax performance was:{}'.format(max(tournament.stats[-1])))
    print('\nThe winning architecture was:{}'.format(tournament.history[-1][np.argmax(tournament.stats[-1])]))
