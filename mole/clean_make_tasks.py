import numpy as np
from scipy.sparse import vstack
import eden_tricks
from graphlearn01 import graphlearn as glearn
from eden.graph import Vectorizer
from eden.util import selection_iterator
from eden_chem.io.pubchem import download
from eden_chem.io.rdkitutils import sdf_to_nx as babel_load
from graphlearn01.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar as grammar
from graphlearn01.learnedlayer import cascade as cascade
from graphlearn01.minor import decompose as decompose
from graphlearn01.minor.molecule import transform_cycle as mole
from sklearn.linear_model import SGDClassifier
from toolz import pipe, curry
from collections import namedtuple
import copy

task = namedtuple("task",'samplerid size repeat sampler neg pos')

###################################
#  DATA [[data p1 with reps][data p2 with repeats]..], oracle
###################################

download_active = curry(download)(active=True,stepsize=50) # default stepsize = 50 (way to few)
download_inactive = curry(download)(active=False,stepsize=50)

def vectorize(thing):
    v = Vectorizer()
    if not thing:
        raise Exception( "need something to vectirize.. received %s" % str(thing))
    thing=list(thing) # current eden does not eat generators anymore? weird
    return v.transform(thing)


def get_data(assay_id):
    active_X = pipe(assay_id, download_active, babel_load, vectorize)
    inactive_X = pipe(assay_id, download_inactive, babel_load, vectorize)
    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    graphs_p = list(pipe(assay_id, download_active, babel_load))
    graphs_n = list(pipe(assay_id, download_inactive, babel_load))


    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)
    print {'active':active_X.shape[0], 'inactive':inactive_X.shape[0]}
    return X, y, graphs_p, graphs_n,esti


def make_data(assay_id,
              repeats=3,
              train_sizes=[50]):

    X,y,graphs_p,graphs_n, esti = get_data(assay_id)

    print 'indicator of tak-ease:'
    print eden_tricks.task_difficulty(X,y)

    for size in train_sizes:
        for repeat in range(repeats):
            poslist = np.random.permutation(range(len(graphs_p)))[:size]
            neglist = np.random.permutation(range(len(graphs_n)))[:size]
            #r={}
            #r['pos']= list(selection_iterator(graphs_p, poslist))
            #r['neg']= list(selection_iterator(graphs_n, neglist))
            neg= list(selection_iterator(graphs_n, neglist))
            pos= list(selection_iterator(graphs_p, poslist))
            for samplerid, sampler in enumerate(make_samplers_chem()):
                yield task(samplerid,size,repeat,sampler,copy.deepcopy(neg),copy.deepcopy(pos))
    yield esti




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

    '''
    mycascade = cascade.Cascade(depth=2,
                          debug=False,
                          multiprocess=True,
                          max_group_size=6,
                          min_group_size=2,
                          num_classes=2)
    '''

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
