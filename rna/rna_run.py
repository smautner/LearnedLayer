import time
from collections import namedtuple

sampled = namedtuple("sampled",'samplerid,size,repeat,time,sequences')
# task = namedtuple("task",'samplerid size repeat sampler sequences')


def run(task):
    print 'runner start,'
    starttime = time.time() 
    thing= list(task.sampler.fit_transform(task.sequences))
    # if sampling stopps at step 0, e[0] does not exist, we filter these with len e > 0
    result = [ e[0] for e in thing if len(e)>0]
    return sampled(task.samplerid,task.size,task.repeat, time.time()-starttime, result)










