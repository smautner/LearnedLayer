'''
i just wrote this a few weeks back,,, but its horrible.. what was i thinking? 
i should rewrite this... 
'''
from toolz import curry, compose, concat, pipe, first, second, take
import time
import matplotlib
matplotlib.use('Agg')
from eden_chem.io.pubchem import download
from eden.graph import Vectorizer
import numpy as np
from scipy.sparse import vstack
from eden_chem.io.rdkitutils import sdf_to_nx as babel_load  # !!!
from eden_chem.display.rdkitutils import nx_to_image
from eden.util import selection_iterator
from graphlearn.trial_samplers import GAT
# DISPLAY IMPORTS
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
#from eden_display import plot_confusion_matrices
#from eden_display import plot_aucs
from sklearn.linear_model import SGDClassifier
import eden_tricks
import graphlearn
import graphlearn.learnedlayer.cascade as cascade
import graphlearn.minor.molecule.transform_cycle as mole
import graphlearn.minor.decompose as decompose
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
download_active = curry(download)(active=True,stepsize=50) # default stepsize = 50 (way to few)
download_inactive = curry(download)(active=False,stepsize=50)


from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=1)


def vectorize(thing):
    v = Vectorizer()
    if not thing:
        raise Exception( "need something to vectirize.. received %s" % str(thing))
    thing=list(thing) # current eden does not eat generators anymore? weird
    return v.transform(thing)

def transpose(things):
    return map(list,zip(*things))

def test(estimator, X):
    '''
    estimator: predict and predict_proba are required
    X: ??? 

    returns:
    [y_pred,y_score]
    '''
    y_pred = estimator.predict(X)
    y_score = estimator.predict_proba(X)[:, 0]
    return [y_pred, y_score]

def error(esti,X,y):
    pred = esti.predict(X) == y
    return float(sum(pred))/ len(y)





'''
There is going to be a main that is

1. make data [(test),test_trained_esti,train_graphs] for each repeat

2. conducting training   train_graphs -> newestisestis, newgraphs

3. evaluate things roc, graph_quality, select graphs
4. draw   roc, quality of graphs, some newgraphs
'''

def get_data(assay_id):
    active_X = pipe(assay_id, download_active, babel_load, vectorize)
    inactive_X = pipe(assay_id, download_inactive, babel_load, vectorize)

    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    graphs = list(pipe(assay_id, download_active, babel_load)) + list(pipe(assay_id, download_inactive, babel_load))
    stats={'active':active_X.shape[0], 'inactive':inactive_X.shape[0]}

    return X, y, graphs, stats


def make_data(assay_id,repeats=3,
              trainclass=1,
              train_sizes=[50],
              not_train_class=-1,
              test_size_per_class=300,
              pick_strategy='random'):
    '''
    :param assay_id:
    :param repeats:
    :param trainclass:
    :param train_size:
    :param not_train_class:
    :param test_size_per_class:
    :param neg_vec_count:
    :param pick_strategy:
            "random"
            "highest scoring "
            "cluster"
    :return:
    [trainsize_i]*repeats
    '''

    #   [(test), test_trained_esti, train_graphs] for each repeat

    X,y,graphs,stats= get_data(assay_id)
    print 'indicator of tak-ease:'
    print eden_tricks.task_difficulty(X,y)
    print stats

    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)

    def get_run( train_size):
        neg_vec_count = train_size
        # get train items
        possible_train_ids = np.where(y == trainclass)[0]

        if pick_strategy=='random':
            train_ids = np.random.permutation(possible_train_ids)[:train_size]
        else:
            # RATE ALL THE GRAPHS, TAKE THE BEST #TRAINSIZE
            #if pick_strategy == highest_scoring
            possible_train_graphs_values = esti.decision_function(X[possible_train_ids])
            train_ids = np.argpartition(possible_train_graphs_values,-train_size)[-train_size:]

            if pick_strategy == 'cluster':
                # CLUSTER THE BEST ONES, USE THE BIGGEST CLUSTER
                n_clusters=3
                clusterer=KMeans(n_clusters=n_clusters)
                res=clusterer.fit_predict(X[train_ids])
                max=0
                id = -1
                for i in range(n_clusters):
                    cnt = np.count_nonzero(res==i)
                    if cnt > max:
                        max=cnt
                        id = i
                    #print "%d %d" % (i, cnt)
                train_ids = train_ids[res==id]
                # this sould be the same as train_ids, so the classes are ballanced...
                neg_vec_count = len(train_ids)


        train_graphs = list(selection_iterator(graphs, train_ids.tolist()))

        # MAKE THE DATA
        possible_test_ids_1 = np.array(list( set(possible_train_ids) - set(train_ids)))
        possible_test_ids_0 = np.where(y == not_train_class)[0]

        test_ids_1 = np.random.permutation(possible_test_ids_1)[:test_size_per_class]
        shuffled_negs = np.random.permutation(possible_test_ids_0)
        test_ids_0 = shuffled_negs[:test_size_per_class]

        test_ids= np.hstack((test_ids_1,test_ids_0))
        X_test = X[test_ids]
        Y_test = y[test_ids]


        neg_vec_ids=shuffled_negs[test_size_per_class:test_size_per_class + neg_vec_count]
        neg_vecs=X[neg_vec_ids]

        #esti= SGDClassifier(loss='log')
        #esti.fit(X_test,Y_test)
        return {'X_test':X_test,'y_test':Y_test,'oracle':esti,'graphs_train':train_graphs,'neg_vecs':neg_vecs}

    return [ [get_run(ts) for ts in train_sizes] for i in range(repeats)]



############################################################################


import numpy as np
import matplotlib.pyplot as plt
def make_inbetween_plot(labels=[50,100,150],means=[(.20, .35, .40),(.20, .40 , .60),(.20,.25,.30)],stds=[(.2, .3,.5),(.3, .3,.3),(.5,.5,.5)],fname='asd.png',dynamic_ylim=False):
    '''
    asdasd 
    '''
    #N = len(labels)
    #ind = np.arange(N) 
    #width = 0.35


    plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(14)
    #ax.ylim(0.0,100)
    if dynamic_ylim:
        plt.ylim=(0, max(means[-1]) + max(stds[-1]))  
    else:
        plt.ylim(-.5,1.5)
    plt.xlim(0,1000)

    
    def fillthing(y,std,label='some label',col='b'):
        y=np.array(y)
        std=np.array(std)
        ax.fill_between(labels,y+std,y-std,facecolor=col,alpha=0.15,linewidth=0)
        #ax.plot(labels,y,label=label,color='gray')
        ax.plot(labels,y,color='gray')
        
    fillthing(means[0],stds[0],col='#6A9AE2')
    fillthing(means[1],stds[1],col='#F94D4D')
    fillthing(means[2],stds[2],col='#555555')

    ax.plot(labels,means[0],label='default',color='b',linewidth=2.0)
    ax.plot(labels,means[1],label='hand',color='r',linewidth=2.0)
    ax.plot(labels,means[2],label='learned',color='black',linewidth=2.0)
    #ax.plot(labels,infernal,label='Infernal',color='#3F3F3F',linewidth=2.0,ls='--')
    #plt.axhline(y=38, color='black',linewidth=2,ls='dotted')
    
    # add some text for labels, title and axes ticks
    labelfs=16
    ax.set_ylabel('score (by oracle)',fontsize=labelfs)
    ax.set_xlabel('number of seeds given',fontsize=labelfs)
    ax.legend(loc='upper left')
    plt.savefig(fname)
    plt.show()



###################################################################


class output_corrected_graphlearn(graphlearn.graphlearn.Sampler):
    def _return_formatter(self, graphlist, mon):
        for e in graphlist:
            yield e


def get_no_abstr(n_jobs=1):
    return output_corrected_graphlearn(n_steps=50,n_jobs=n_jobs)

def get_hand_abstr(n_jobs=1):
    return output_corrected_graphlearn(n_jobs=n_jobs,
    select_cip_max_tries=100,
    size_constrained_core_choice=5,
            # i changed the defaults for the strategy... it seems that
            # 1. size constraint is not combinable with the other chip choice plans
            # 2. size constraint core choice reduces the error rate compared to by_frequency, (probably)
    decomposer= decompose.MinorDecomposer(),
    graphtransformer= mole.GraphTransformerCircles())
    

def get_casc_abstr(n_jobs=1):
    mycascade = cascade.Cascade(depth=2,
                          debug=False,
                          multiprocess=True,
                          max_group_size=6,
                          min_group_size=2,
                          num_classes=2)

    return output_corrected_graphlearn(n_jobs=n_jobs,
    select_cip_max_tries=100,
    size_constrained_core_choice=5,
    decomposer= decompose.MinorDecomposer(),
    graphtransformer= mycascade)

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

###################################################################

def runwrap(sampler,graphs,attempt=0):
    start=time.time()
    
    try:
        graphs=list(sampler.fit_transform(graphs))
    except ValueError:
        # this happens when name_estimator does not have enough subgraphs extracted to train
        # the nn:    clusterclassifier: fit: neigh.fit(data)
        if attempt < 3:
            
            return runwrap(sampler,graphs,attempt+1)
        else:
            
            print 'runwrap failed, retrying! graphs#%d' % len(graphs)
            raise Exception("attept 3... there were %d graphs" % len(graphs))
    if not graphs:
        print "runwrap_no_results"
        exit()
    timeused = time.time()- start
    return (graphs,timeused)

def run_experiments(samplers, data):
    # this is bypassing the time annotation:
    #return  [[[  list(s.fit_transform(problem_dict['graphs_train']))
    return  [[[ runwrap(s,problem_dict['graphs_train']) 
                     for problem_dict in repeat ]
                    for repeat in data]
                   for s in samplers]


    
def evaluate(graphs, task_data):
    means = []
    stds = []
    means_time=[]
    stds_time=[]
    # evaluate results...
    for s in graphs: # s stands for sampler.. its the result for a single sampla
        # error vectorize, test
        data = [[ (test( task_data[0][0]['oracle'], vectorize(outgraphs))[1],time)
                  for outgraphs,time in repeats ]
                for repeats in s]
        # they are ordered by repeats now.
        data = transpose(data)
        # now they are ordered by seedcount :)
        res=[]
        res_time=[]
        for row in data:
            row,time_row= transpose(row)
            score_row = [thing for l2 in row for thing in l2] # flatten

            res.append([np.mean(score_row),np.std(score_row)])
            res_time.append([np.mean(time_row),np.std(time_row)])
        res=transpose(res)
        res_time=transpose(res_time)

        means.append(res[0])
        stds.append(res[1])
        means_time.append(res_time[0])
        stds_time.append(res_time[1])
    return means,stds, means_time, stds_time


# make a data source
assay_id = '624466'  # apr88
assay_id = '588350'  # apr86
assay_id = '449764'  # apr85
assay_id = '492952'  # apr85
assay_id = '463112'  # apr82
assay_id = '463213'  # apr70
assay_id = '119'  # apr60 30k mols
assay_id = '1224857'  # apr10
assay_id = '2326'  # apr03 200k mols
assay_id = '1834'  # apr90 500 mols



assay_id = '651610'  # apr93 23k mols
repeats = 3
train_sizes = [20,50,100,200,500,750,1000]
n_jobs=4

'''
THE PLAN IS SIMPLE
1. make_samplers  ( there are 3 )
2. evaluate results  -> mmm sss mmm sss
3. probably transpose
4. draw

'''

if True:  # debug
    assay_id = '1834' # 1834 is bad because there are too few compounds :D  65* is too large for testing
    repeats = 2
    train_sizes= [20,50]
    n_jobs=1

if __name__ == '__main__':

    if True:
        samplers_chem = make_samplers_chem(n_jobs=n_jobs)
        if False: # NEW TEST JUN 17!!
            samplers_chem= [samplers_chem[1]]
            repeats  = 1
            train_sizes=[1000,1000]

        data_chem  = make_data(assay_id,
                       repeats=repeats,
                       trainclass=1,
                       train_sizes=train_sizes,
                       test_size_per_class=300,
                       pick_strategy='cluster') # cluster random  highscoring

        import dill 
        dill.dump(samplers_chem, open("samplers", "w"))
        dill.dump(data_chem, open("data", "w"))
        exit()




        graphs_chem = run_experiments(samplers_chem,data_chem)
        means,stds,means_time, stds_time = evaluate(graphs_chem,data_chem)
        #s="""
        #from layerutils import make_inbetween_plot
        #make_inbetween_plot(labels=%s,means=%s,stds=%s) 
        #""" % (str(train_sizes),str(means), str(stds))

        #with open("sampler1ok.py","w") as f:
        #    f.write(s)

        make_inbetween_plot(labels=train_sizes, means=means , stds=stds,fname='chem.png')
        make_inbetween_plot(labels=train_sizes, means=means_time, stds=stds_time,fname='chem_time.png',dynamic_ylim=True)
        
        # NOTES:
        # 1. there are a few ascii draws in gl: cascade.py and abstractor.py
        # 2. i inserted an exit in this py file





###############################
# STUFF BELOW HERE IS JUST leftovers.. 
#######################################################################


'''
def generative_training_2(data,niter,get_sampler=GAT.get_sampler):
    # this is the version that uses real negs :)

    # data -> [estis]*niter, [gengraphs]*niter
    # this is the version with the real negatives
    train= lambda x,y: GAT.generative_adersarial_training_HACK(
        get_sampler(), n_iterations=niter, seedgraphs=x, neg_vectors=y,partial_estimator=False)

    stuff = [ train(x['graphs_train'], x['neg_vecs']) for x in data ]
    return transpose(stuff)

def generative_training(data,niter,get_sampler = GAT.get_sampler):
    # data -> [estis]*niter, [gengraphs]*niter
    train= lambda x: GAT.generative_adersarial_training(
        get_sampler(), n_iterations=niter, seedgraphs=x, partial_estimator=False)
    stuff = [ train(x['graphs_train']) for x in data ]
    return transpose(stuff)

#########################################################################


def roc_data(estis,data):
    # test each generated estimator with the according estimator
    dat = [ map(lambda x: [dat['y_test']]+test(x,dat['X_test']) ,esti ) for esti,dat in zip(estis,data)]
    # transposing orders the result by level or depth or whatever.
    # the map above is generating a 2d matrix -> we have $repeats many
    # we can just hstack the 2d arrays..
    return [np.hstack(tuple(allrepeats)) for allrepeats in transpose(dat)]

def drawroc_data(rocdata):
    for e in rocdata:
        predicitve_performance_report(e)
def predicitve_performance_report(data, filename=None):
    y_true, y_pred, y_score = data
    line_size = 135
    print '_' * line_size
    print
    print 'Accuracy: %.2f' % accuracy_score(y_true, y_pred)
    print ' AUC ROC: %.2f' % roc_auc_score(y_true, y_score)
    print '  AUC AP: %.2f' % average_precision_score(y_true, y_score)
    print '_' * line_size
    print
    print 'Classification Report:'
    print classification_report(y_true, y_pred)
    print '_' * line_size
    print
    if filename:
        conf_filename = 'confus_%d_.png' % filename
        auc_filename = 'auc_%d_.png' % filename
    else:
        conf_filename, auc_filename = None, None
    plot_confusion_matrices(y_true, y_pred, size=int(len(set(y_true)) * 2.5), filename=conf_filename)
    print '_' * line_size
    print
    plot_aucs(y_true, y_score, size=10, filename=auc_filename)
##
def select_graphs(graphs,estis, print_best=5 ):
    if print_best > 0:
        # calculate the scores of the graphs  with the right estimators... GATdepth*repeats
        scores = [[e.predict(vectorize(g)) for g,e in zip(gs,es)]  for gs,es in zip(graphs,estis)]
        # take the graphs with the best scores ..
        graphs = [[ list(selection_iterator(gr,np.argpartition(sco,-print_best)[-print_best:]))
                    for gr,sco in zip(grs,scores)]   for grs,scores in zip(graphs,scores)]
        # collapse graphs that are on the same GAT-level
        return map(lambda x: reduce(lambda y,z: z+y,x),transpose(graphs))




from graphlearn.utils import molecule
def draw_select_graphs(graphs):
    for i, graphlist in enumerate(graphs):
        print 'best graphs (according to GAT) for repeat #%d' % i
        molecule.draw(graphlist)
        #pic = nx_to_image(graphlist)
        #plt.figure()
        #plt.imshow(np.asarray(pic))

##
def test_vs_gatesties(estis,data):

    scores = [[error(level_esti,repeats_data['X_test'],repeats_data['y_test'])# level_est and repeats_data
               for level_esti in repeats_es]
                    for repeats_data, repeats_es in zip(data, estis)]

    # scores are now [l1,l2,l3..][l1,l2,l3..][l1,l2,l3..]
    # scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    #print scores
    scores = transpose(scores)
    # transpose to get: [l1,l1,l1..][l2,l2,l2..]  .. and we just flatten the [l_x,l_x..]
    return transpose([ [np.mean(e),np.std(e)] for e in scores])

##
def internalgat(estis, newgraphs):
    scores = [[level_esti.predict_proba(vectorize(level))[:,0] for level,level_esti in zip(repeats,repeats_es[1:])] for repeats, repeats_es in zip(newgraphs, estis)]
    # scores are now [l1,l2,l3..][l1,l2,l3..][l1,l2,l3..]
    scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    # transpose to get: [l1,l1,l1..][l2,l2,l2..]  .. and we just flatten the [l_x,l_x..]
    return transpose([ [np.mean(e),np.std(e)] for e in scores])

##
def graphlol(data, newgraphs):
    estis= [d['oracle'] for d in data]
    scores = [[es.predict_proba(vectorize(level))[:,0] for level in repeats] for repeats, es in zip(newgraphs, estis)]
    # order by level and concatenate over all repeats
    scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    # get means snd stds
    return transpose([ [np.mean(e),np.std(e)] for e in scores])




def draw_graph_quality(data):
    means,stds= data
    plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(14)
    #plt.ylim(0, 80)
    #plt.xlim(0, 400)
    plt.axhline(y=38, color='black', linewidth=3)

    def fillthing(y, std, label='some label', col='b'):
        y = np.array(y)
        std = np.array(std)
        ax.fill_between('asd', y + std, y - std, facecolor=col, alpha=0.3, linewidth=0)
        # ax.plot(labels,y,label=label,color='gray')
        ax.plot('asd', y, color='gray')

    fillthing(means, stds, col='#6A9AE2')

    ax.plot('asdasd', means, label='new CIP', color='b', linewidth=2.0)
    # add some text for labels, title and axes ticks
    labelfs = 16
    ax.set_ylabel('something', fontsize=labelfs)
    ax.set_xlabel('something2', fontsize=labelfs)
    ax.legend(loc='lower right')
    plt.show()

def simple_draw_graph_quality(data,title='title',file=None,ylabel='Accuracy', xlabel="GAT Iteration"):
    means,std = data
    plt.figure()

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.errorbar(range(len(means)), means,  yerr=std)
    plt.title(title)

    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

##


def evaluate_all(data,estis,newgraphs,draw_best=5):
    rocdata = roc_data(estis,data)            # what does this need to look like?
    graphs =  select_graphs(newgraphs, estis, print_best=draw_best )      # select some graphs that need drawing later
    graph_quality = graphlol(data,newgraphs)
    graph_quality_internal = internalgat(estis,newgraphs)
    res5 = test_vs_gatesties(estis,data)
    return rocdata,graphs,graph_quality, graph_quality_internal, res5



0
0'''
######################################################################


