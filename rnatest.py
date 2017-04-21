
from rnalayer import get_data2
from layerutils_rna import make_samplers_rna

from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=1)




def run_experiments(data,samplers):

    #1  one is the forgi
    #2  two is the learned guy 
    samplers=[samplers[1]]

    return [[[[ list(sampler.fit_transform (problem))]
        for problem in repeat]
            for repeat in  data] 
                for sampler in samplers]

data= get_data2('RF00005',repeats=2, trainsizes=[10,20])
samplers = make_samplers_rna(n_jobs=4)

print run_experiments(data,samplers)



