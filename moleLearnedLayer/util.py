from eden_chem.io.pubchem import download
from eden_chem.io.rdkitutils import sdf_to_nx
from eden.graph import Vectorizer
from graphlearn01 import graphlearn as glearn
from graphlearn01.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar as grammar
from graphlearn01.learnedlayer import cascade as cascade
from graphlearn01.minor import decompose as decompose
from graphlearn01.minor.molecule import transform_cycle as mole
from graphlearn01.utils import draw
from graphlearn01 import estimate as glesti
from collections import namedtuple
from scipy.sparse import vstack
from copy import deepcopy
from sklearn.linear_model import SGDClassifier
import time
import random
from toolz import curry, pipe
import dill
from eden.io import gspan
import numpy as np
import sklearn

def vectorize(instances):
    vec=Vectorizer()
    return vec._transform_serial(instances)

#####
# structs
####
sampled = namedtuple("sampled",'samplerid,size,repeat,time, graphs')
task = namedtuple("task",'samplerid size repeat sampler neg pos')
task2 = namedtuple("task2",'samplerid size repeat sampler neg pos negtest postest')
processed_result=namedtuple("processed_result","samplerid, size, score_mean, score_var, time_mean, time_var, score_sub_means")


#########
# generic  graph things
#########
dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "w"))
loadfile = lambda filename: dill.load(open(filename, "r"))

load_nx_dumps = lambda:  (loadfile("pos.nxdump"), loadfile("neg.nxdump"))

def getgraphs(aid):
    if aid == 'bursi':
        return list(gspan.gspan_to_eden("bursi.pos.gspan")), list(gspan.gspan_to_eden("bursi.neg.gspan"))
    download_active = curry(download)(active=True,stepsize=50)
    download_inactive = curry(download)(active=False,stepsize=50)
    active = pipe(aid, download_active, sdf_to_nx,list)
    inactive = pipe(aid, download_inactive, sdf_to_nx, list)
    return active,inactive

def sample_pos_neg(graphs_pos,graphs_neg, size_pos=100, size_neg=100, repeats=1):
    makeset= lambda x:  (random.sample(graphs_pos,size_pos), random.sample(graphs_neg,size_neg))
    return map(makeset,range(repeats))

def sample_pos_neg_no_duplicates(pos,neg,size,repeats):
    res=[]
    for i in range(repeats):
        random.shuffle(pos)
        random.shuffle(neg)
        res.append((pos[:size],neg[:size]))
        pos=pos[size:]
        neg=neg[size:]
    return res

def aid_to_linmodel(aid):
    return graphs_to_linmodel( *getgraphs(aid) )

def graphs_to_Xy(pos,neg):
    active_X = vectorize(pos)
    inactive_X = vectorize(neg)
    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    return X,y

def graphs_to_linmodel(pos,neg):
    X,y = graphs_to_Xy(pos,neg)
    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)
    return esti

def graphs_to_acc(g1,g2,g3,g4):
    '''
    train pos, train neg, test pos, test neg
    '''
    esti=graphs_to_linmodel(g1,g2)
    X,y= graphs_to_Xy(g1,g2)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)
    return acc



#######
# optimisation
#######
def init_optimisation(aid='1834',size=100,repeats=3, dump=True):
    ''' dumps [[possizeGraphs,negsizeGraphs] * repeats] into a file and returns fname '''
    pos,neg = getgraphs(aid)

    stack_of_sampled_graphsets = sample_pos_neg_ESTI(pos,neg,size,size,repeats)

    if dump:
        name = 'task_%s_%d_%d_%d' % (aid,size,size,repeats)
        dumpfile(stack_of_sampled_graphsets,name)
        return name
    else:
        return stack_of_sampled_graphsets


def sample_pos_neg_ESTI(graphs_pos,graphs_neg, size_pos=100, size_neg=100, repeats=1):
    def makeset(a):
        random.shuffle(graphs_pos)
        random.shuffle(graphs_neg)
        return (graphs_pos[:size_pos] , graphs_neg[:size_neg]) , graphs_to_linmodel( graphs_pos[size_pos:], graphs_neg[size_neg:])
    return zip(* map(makeset,range(repeats)))


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

def get_all_samplers(n_jobs=1,params=[{},{},{}],select=[0,1,2]):
    '''
    :return:
     all 3 samplers have a fit_transform(graphs),....
     when it comes to sampling given 2 classes, there needs to be more work :)
    '''
    samplers=[get_no_abstr(n_jobs=n_jobs,kwargs=params[0]),
              get_casc_abstr(n_jobs=n_jobs,kwargs=params[1]),
              get_hand_abstr(n_jobs=n_jobs,kwargs=params[2])]

    samplers=[samplers[i] for i in select]
    #samplers=[get_casc_abstr() for i in range(3)]
    #print 'samplers are fake atm'
    return samplers


def sample(task, debug_fit=False,skipgrammar=False):
    start=time.time()
    # make pos/neg decomposers
    numpos=len(task.pos)
    decomposers = [task.sampler.decomposer.make_new_decomposer(data)
                     for data in task.sampler.graph_transformer.fit_transform(task.pos,task.neg)]

    if skipgrammar:
        return [d._unaltered_graph for d in decomposers]

    # fit grammar
    task.sampler.fit_grammar(decomposers)
    # fit estimator
    task.sampler.estimator= glesti.TwoClassEstimator()
    task.sampler.fit_estimator(decomposers[:numpos],negative_decomposers=decomposers[numpos:])


    if debug_fit: # we do this after the fit esti, so we can also see of the fitting crashes
        draw.draw_grammar(task.sampler.lsgg.productions, abstract_interface=True, n_graphs_per_line=7, n_productions=5, n_graphs_per_production=7)
        return
    # run
    graphs=  list(task.sampler.transform(task.pos))
    return sampled(task.samplerid,task.size,task.repeat,time.time()-start,graphs)





def quickfit(aid,size,params, skipgrammar=False):
    sampler = get_casc_abstr(kwargs=params)
    if aid=='load_nx_dumps':
        po,ne= load_nx_dumps()
    else:
        p,n  = getgraphs(aid)
        po, ne = sample_pos_neg(p,n,size_pos=size,size_neg=size, repeats=1)[0]
    return sample( task(1,size,0,sampler,ne,po) ,debug_fit=True, skipgrammar=skipgrammar)




def graphs_to_scores(graphs, oracle):
    graphs = vectorize(graphs)
    scores = oracle.decision_function(graphs)
    return scores



def bursi_get_extremes(num=200):
    po,ne = list(gspan.gspan_to_eden("bursi.pos.gspan")), list(gspan.gspan_to_eden("bursi.neg.gspan"))
    X,y = graphs_to_Xy(po,ne)
    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)
    res= [ (score,idd) for idd, score in enumerate(esti.decision_function(X))] # list
    res.sort()
    graphs=po+ne
    # returns pos/neg
    return [graphs[idd] for (score,idd) in res[0-num:]], [graphs[idd] for (score,idd) in res[:num] ]





