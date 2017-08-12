from eden_chem.io.pubchem import download
from eden_chem.io.rdkitutils import sdf_to_nx
from eden.graph import vectorize
from graphlearn01 import graphlearn as glearn
from graphlearn01.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar as grammar
from graphlearn01.learnedlayer import cascade as cascade
from graphlearn01.minor import decompose as decompose
from graphlearn01.minor.molecule import transform_cycle as mole
from graphlearn01 import estimate as glesti
from collections import namedtuple
from scipy.sparse import vstack
from copy import deepcopy
from sklearn.linear_model import SGDClassifier
import time
import random
from toolz import curry, pipe
import dill
import numpy as np


#####
# structs
####
sampled = namedtuple("sampled",'samplerid,size,repeat,time, graphs')
task = namedtuple("task",'samplerid size repeat sampler neg pos')

#########
# generic  graph things
#########
dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "w"))
loadfile = lambda filename: dill.load(open(filename, "r"))

def getgraphs(aid):
    download_active = curry(download)(active=True,stepsize=50)
    download_inactive = curry(download)(active=False,stepsize=50)
    active = pipe(aid, download_active, sdf_to_nx,list)
    inactive = pipe(aid, download_inactive, sdf_to_nx, list)
    return active,inactive


def sample_pos_neg(graphs_pos,graphs_neg, size_pos=100, size_neg=100, repeats=1):
    makeset= lambda x:  (random.sample(graphs_pos,size_pos), random.sample(graphs_neg,size_neg))
    return map(makeset,range(repeats))

def aid_to_linmodel(aid):
    return graphs_to_linmodel( *getgraphs(aid) )

def graphs_to_linmodel(pos,neg):
    active_X = vectorize(pos)
    inactive_X = vectorize(neg)
    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)
    return esti



#######
# optimisation
#######
def init_optimisation(aid='1834',size_pos=100,size_neg=100,repeats=3, dump=True):
    ''' dumps [[possizeGraphs,negsizeGraphs] * repeats] into a file and returns fname '''
    pos,neg = getgraphs(aid)
    tasks = sample_pos_neg(pos,neg,size_pos,size_neg,repeats)
    if dump:
        name = 'task_%s_%d_%d_%d' % (aid,size_pos,size_neg,repeats)
        dumpfile(tasks,name)
        return name
    else:
        return tasks


######################################
#  SAMPLERS
######################################

class output_corrected_graphlearn(glearn.Sampler):
    def _return_formatter(self, graphlist, mon):
        for e in graphlist:
            yield e


def get_no_abstr(n_jobs=1, kwargs={"n_steps":50}):
    kwargs=kwargs.copy()
    grammaropts= kwargs.get('grammar_options',{})
    kwargs.pop("grammar_options",None)
    return output_corrected_graphlearn(
        n_jobs=n_jobs,
        grammar=grammar(**grammaropts),**kwargs)


def get_hand_abstr(n_jobs=1,kwargs={
    "select_cip_max_tries":30,
    "size_constrained_core_choice":5, }):

    kwargs=kwargs.copy()
    grammaropts= kwargs.get('grammar_options',{})
    kwargs.pop("grammar_options",None)

    return output_corrected_graphlearn(n_jobs=n_jobs,
        grammar=grammar(**grammaropts),
        decomposer= decompose.MinorDecomposer(),
        graphtransformer= mole.GraphTransformerCircles(),
        **kwargs)


def get_casc_abstr(n_jobs=1, kwargs={
    "select_cip_max_tries":30,
    "size_constrained_core_choice":5, }):

    kwargs=kwargs.copy()
    grammarargs=kwargs.pop("grammar_options",{})
    learnargs=kwargs.pop("learn_params",{})


    mycascade = cascade.Cascade(**learnargs)


    return output_corrected_graphlearn(n_jobs=n_jobs,
        decomposer= decompose.MinorDecomposer(),
        grammar=grammar(**grammarargs),
        graphtransformer= mycascade,**kwargs)

def make_samplers_chem(n_jobs=1):
    '''
    :return:
     all 3 samplers have a fit_transform(graphs),....
     when it comes to sampling given 2 classes, there needs to be more work :)
    '''
    samplers=[get_no_abstr(n_jobs=n_jobs),get_hand_abstr(n_jobs=n_jobs),get_casc_abstr(n_jobs=n_jobs)]
    #samplers=[get_casc_abstr() for i in range(3)]
    #print 'samplers are fake atm'
    return samplers


def sample(task):
    start=time.time()
    # make pos/neg decomposers
    decomposers_n = [task.sampler.decomposer.make_new_decomposer(data)
                     for data in task.sampler.graph_transformer.fit_transform(task.neg)]
    decomposers_p = [task.sampler.decomposer.make_new_decomposer(data)
                   for data in task.sampler.graph_transformer.fit_transform(deepcopy(task.pos))]
    # fit grammar
    task.sampler.fit_grammar(decomposers_p)

    # fit estimator
    task.sampler.estimator= glesti.TwoClassEstimator()
    task.sampler.fit_estimator(decomposers_p,negative_decomposers=decomposers_n)
    # run
    graphs=  list(task.sampler.transform(task.pos))
    timeused = time.time()- start
    return sampled(task.samplerid,task.size,task.repeat,timeused,graphs)

