'''
i just wrote this a few weeks back,,, but its horrible.. what was i thinking? 
i should rewrite this... 
'''
import matplotlib
import time

from moleLearnedLayer.clean_make_tasks import vectorize, make_data, make_samplers_chem

matplotlib.use('Agg')
# DISPLAY IMPORTS
#from eden_display import plot_confusion_matrices
#from eden_display import plot_aucs

from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=1)




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


class draw:
    def __init__(self, xlabels, dynamic_y):
        # plt clear ! however that works..
        self.xlabels=xlabels
        self.dynamic_y = dynamic_y

    def add_line(self,means,variances):


def make_inbetween_plot(labels=[50,100,150],
                        means=[(.20, .35, .40),(.20, .40 , .60),(.20,.25,.30)],
                        stds=[(.2, .3,.5),(.3, .3,.3),(.5,.5,.5)],
                        fname='asd.png',
                        dynamic_ylim=False):
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
        data = [[ (test(task_data[0][0]['oracle'], vectorize(outgraphs))[1], time)
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

def evaluate2(num_rep_gt, oracle, time=False):
    '''
    plan is to just return an iterator over data points..
    so a call to evaluate produces exactly 1 line

    also task data musst be selcted carefully
    '''

    for num in num_rep_gt:
        interest=[ repeat[time] for repeat in num]
        if time:
            i=np.array(interest)
            yield i.mean(), i.var()
        else:
            # interest should be a list of list of graphs
            interest=[ e for e in lisd  for lisd in interest ]
            interest= vectorize(interest)
            res = test(oracle,interest)
            r1,r2 = zip(*res) # which is right?

            print r1,r2
            exit()

            yield r1.mean(), r1.var()
            yield r2.mean(), r2.var()





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

if False:  # debug
    assay_id = '1834' # 1834 is bad because there are too few compounds :D  65* is too large for testing
    repeats = 2
    train_sizes= [20,50]
    n_jobs=1

if __name__ == '__main__':

    if True:
        samplers_chem = make_samplers_chem(n_jobs=n_jobs)

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



