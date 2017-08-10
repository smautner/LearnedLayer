import time
from graphlearn01 import estimate as glesti
from collections import namedtuple

#task = namedtuple("task",'samplerid size repeat sampler neg pos')

sampled = namedtuple("sampled",'samplerid,size,repeat,time,graphs')

def runwrap(task,attempt=0):

    start=time.time()

    # make pos/neg decomposers
    decomposers_n = [task.sampler.decomposer.make_new_decomposer(data)
                     for data in task.sampler.graph_transformer.fit_transform(task.neg)]
    decomposers_p = [task.sampler.decomposer.make_new_decomposer(data)
                   for data in task.sampler.graph_transformer.fit_transform(task.pos)]
    # fit grammar
    task.sampler.fit_grammar(decomposers_p)

    # fit estimator
    task.sampler.estimator= glesti.TwoClassEstimator()
    task.sampler.fit_estimator(decomposers_p,negative_decomposers=decomposers_n)
    # run
    graphs=  list(task.sampler.transform(task.pos))


    timeused = time.time()- start


    return sampled(task.samplerid,task.size,task.repeat,timeused,graphs)


