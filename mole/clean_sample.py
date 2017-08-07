import time
from graphlearn01 import estimate as glesti
from collections import namedtuple

#task = namedtuple("task",'samplerid size repeat sampler neg pos')

sampled = namedtuple("sampled",'samplerid,size,repeat,time,graphs')

def runwrap(task,attempt=0):

    start=time.time()

    try:
        # make pos/neg decomposers
        decomposers_p = [task.sampler.decomposer.make_new_decomposer(data)
                       for data in task.sampler.graph_transformer.fit_transform(task.pos)]
        decomposers_n = [task.sampler.decomposer.make_new_decomposer(data)
                         for data in task.sampler.graph_transformer.fit_transform(task.neg)]
        # fit grammar
        task.sampler.fit_grammar(decomposers_p)

        # fit estimator
        task.sampler.estimator= glesti.TwoClassEstimator()
        task.sampler.fit_estimator(decomposers_p,negative_decomposers=decomposers_n)
        # run
        graphs=  list(task.sampler.fit_transform(task.pos))

    except ValueError:
        # this happens when name_estimator does not have enough subgraphs extracted to train
        # the nn:    clusterclassifier: fit: neigh.fit(data)
        if attempt < 3:
            return runwrap(task,attempt+1)
        else:
            print 'runwrap failed, retrying! graphs#%d' % len(graphs)
            raise Exception("attept 3... there were %d graphs" % len(graphs))
    if not graphs:
        print "runwrap_no_results"
        exit()
    timeused = time.time()- start


    return sampled(task.samplerid,task.size,task.repeat,timeused,graphs)


