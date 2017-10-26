import numpy as np
import pprint
import util
from graphlearn01.learnedlayer.cascade import Cascade
import copy
import random
from multiprocessing import Pool
import eden
from graphlearn01.utils import draw
import sklearn
import optimize




# HERE IS THE NEW FABI PLAN

def maketasks(aid, size, test,repeats,filename):
    pos,neg= util.getgraphs(aid)
    repeatsXposnegsamples = util.sample_pos_neg(pos,neg,size+test,size+test,repeats)
    repeatsXposnegsamples = [ [pos2[:size],neg2[:size],pos2[size:], neg2[size:] ]  for pos2,neg2 in repeatsXposnegsamples]
    util.dumpfile(repeatsXposnegsamples,filename)



def getparams_best(): # high scoring connected components
    return {     'dbscan_range': random.uniform(.5,.7),
                 'depth': random.randint(1,4),
                 'subgraphextraction': 'best',
                 'group_score_threshold': random.randint(50,90)/100.0,
                 'clusterclassifier' : random.choice(["keep",'nokeep']),
                 'max_group_size': random.randint(6,10),
                 'min_group_size': random.randint(2,5)}



def getparams_best_interface(): # bet and use the interface stuff
    return {     'dbscan_range': random.uniform(.5,.7),
                 'depth': random.randint(1,4),
                 'clusterclassifier':"interface_keep",
                 'subgraphextraction':"best_interface",
                 'group_score_threshold': random.randint(50,90)/100.0,
                 'min_clustersize': random.randint(2,15),  # this is an exclusive parameter :)
                 'max_group_size': random.randint(6,10),
                 'min_group_size': random.randint(2,5)}


def getparams_best_iftrick(): # best, iftrick
    return {
                        'subgraphextraction':"best_interface",
                        'clusterclassifier':"interface_nocluster",
                        'depth': random.randint(1,4),
                        'min_clustersize': random.randint(5,25),  # this is an exclusive parameter :)
                        'group_score_threshold': random.randint(50,90)/100.0,
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}




def getparams_cutter_interface():  # cutter interface
    return {            'subgraphextraction':"cut_interface",
                        'clusterclassifier':"lotsofclassifiers",
                        'dbscan_range': random.uniform(.4,.6),
                        'depth': random.randint(1,4),
                        'group_score_threshold': random.randint(3,17)/100.0,
                        'min_clustersize': random.randint(10,100)/1000.0,  # this is an exclusive parameter :)
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}





def getparams_cutter_iftrick():
    return {            'subgraphextraction':"cut_interface",
                        'clusterclassifier':"interface_nocluster",
                        'depth': random.randint(1,4),
                        'group_score_threshold': random.randint(3,17)/100.0,
                        'min_clustersize': random.randint(10,100)/1000.0,  # this is an exclusive parameter :)
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}



def getparams_cutter():
    return {            'subgraphextraction':"cut",
                        'clusterclassifier' : "keep",
                        'group_score_threshold': random.randint(3,17)/100.0,
                        'dbscan_range': random.uniform(.5,.7),
                        'depth': random.randint(2,3),
                        'min_clustersize': random.randint(10,100)/1000.0,  # this is an exclusive parameter :)
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}






def gget_special():
    d={
        'subgraphextraction':random.choice( ["cut","best_interface" , 'best', 'cut_interface'])
    }

    if 'best' in d['subgraphextraction']:
        d['group_score_threshold']=  random.randint(50,90)/100.0
    else:
        d['group_score_threshold'] = random.randint(3,17)/100.0

    if 'interface' in d['subgraphextraction']:
        d['clusterclassifier'] = random.choice(["interface_nocluster",'interface_keep'])
    else:
        d['clusterclassifier'] = random.choice(["keep",'nokeep'])
    return d



def gget_basic_params():
    return {            'dbscan_range': random.uniform(.5,.7),
                        'depth': random.randint(1,4),
                        'min_clustersize': random.randint(5,100)/1000.0,  # this is an exclusive parameter :)
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}

def getanyparams():

    params= gget_basic_params()
    params.update(gget_special())
    #prog = random.choice([getparams_best,getparams_best_iftrick, getparams_best_interface, getparams_cutter, getparams_cutter_iftrick, getparams_cutter_interface])
    return params





def getacc( pos,neg, post,negt ):
    esti= util.graphs_to_linmodel(pos,neg)
    X,y = util.graphs_to_Xy(post,negt)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)
    return acc




def getsamplerparam(param):
    paramz=optimize.get_default_samplers_params()
    paramz['learn_params']=param
    return paramz

def getsampler(param):
    paramz=getsamplerparam(param)
    return util.get_casc_abstr(kwargs=paramz)



def run(filename, taskid , getparams="ERROR TERROR", debug=False):
    tasks = util.loadfile(filename)

    if getparams == 'all':
        param = getanyparams()
    else:
        param = eval("getparams_"+getparams+"()")

    scores=[]
    scoreinfo=[]
    for a in tasks:
        sampler=getsampler(param)


        pos,neg, testpos,testneg =  a

        alltrain_composer = [sampler.decomposer.make_new_decomposer(data)
                   for data in sampler.graph_transformer.fit_transform(pos,neg,
                        remove_intermediary_layers=False)]


        alltrain = map(lambda x : x.pre_vectorizer_graph(), alltrain_composer)

        alltest_composer = [sampler.decomposer.make_new_decomposer(data)
                   for data in sampler.graph_transformer.transform(testpos+testneg,remove_intermediary_layers=False)]

        alltest = map(lambda x : x.pre_vectorizer_graph(),alltest_composer)



        s= len(pos)
        ss=len(testpos)
        score =  getacc( alltrain[:s],alltrain[s:],alltest[:ss],alltest[ss:])
        score2 = util.get_compression(alltest_composer)
        score3 = util.get_compression(alltrain_composer)
        if debug:
            from graphlearn01.utils import draw
            print param
            print "press: %.2f   score: %.2f " % (score2,score)
            for e in [2,30,5]:
                layers = alltest_composer[e].get_layers()
                print "graph sizes"+ "\t".join([str(len(z)) for z in layers])
                draw.graphlearn(layers)

        alpha = .9
        scores.append(alpha*score+(1-alpha)*score2)


        train_acc = getacc(  alltrain[:s],alltrain[s:],alltrain[:s],alltrain[s:] )
        crossval = sklearn.model_selection.cross_val_score(sklearn.linear_model.SGDClassifier(loss='log'), *util.graphs_to_Xy(alltrain[:s],alltrain[s:])).mean()

        scoreinfo.append( {"accuracy":score,"accuracy_train":train_acc,"crossval":crossval, "compression_test":score2 , 'compression_train':score3} )

    dumpdata= np.median(scores),param,scoreinfo
    if debug:
        import pprint
        pprint.pprint(dumpdata)
        print "#"*80
    util.dumpfile(dumpdata,"oap/%s_%d" % (filename,taskid))

    #util.dumpfile(  (np.median(scores_debug), np.median(crossvalscores)),"oap/%s_%d_debug_info" % (filename,taskid))



import os
def loadstuff(filename, tasknum):
    for i in range(tasknum):

        fname = "oap/%s_%d" % (filename,i)
        if os.path.exists(fname):
            yield util.loadfile(fname)


        else:
            print "failed to load: %d" % i

def eva(filename, tasknum):
    data = list(loadstuff(filename,tasknum))


    data.sort(reverse=True)

    util.dumpfile(data[:5],'oap/top5_%s' % filename)
    pprint.pprint(  optimize.merge_dicts([ a[1] for a in data[:5] ]))

    print "score_median", [a[0] for a in data[:5]]
    print "accuracy_train_median", [a[2] for a in data[:5]]
    print "crossval_median", [a[3] for a in data[:5]]
    print "acc and compression", [a[4] for a in data[:5]]


def show_best(aid,size=200):
    filename = "top5_%s_oap_task" % aid

    params = util.loadfile(filename)
    for e in params: print e

    for param in params:
        param[1]['debug']=True
        runparam=getsamplerparam(param[1])
        util.quickfit(aid, size, runparam, skipgrammar=True)



if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(  sys.argv[2] , int(sys.argv[3])-1 , sys.argv[1])

# BELOW SHOULD BE JUST CRAP
############################################
#######################################################3

def get_graph_stack(aid="651610", overeight_graph_num=600):
    a,b=util.getgraphs(aid)
    return util.sample_pos_neg_no_duplicates(a,b,overeight_graph_num,4)


def optimize_old(graphs, depth,graph_num,num_tries_per_depth_level=30):
    casc = Cascade(depth=depth,debug=False,max_group_size=20)
    casc.setup_transformers()
    # optimize each level of the cascade
    for e in range(depth):
        best_solution=(None,-1)
        trans = casc.transformers[e]
        trans.prepfit()


        if False:
            for attempt in range(num_tries_per_depth_level):
                solution = test_repeats(trans, graphs, repeats=5)
                if solution[1] > best_solution[1]:
                    best_solution=solution
        else:
            p=Pool(25)
            res= [ eden.apply_async(p,test_repeats,(trans,graphs,2) ) for i in range(num_tries_per_depth_level) ] # solution = test_repeats(trans, graphs, repeats=5)
            p.close()
            for r in res:
                solution=r.get()

                print "there was a test",solution[0].score_threshold, solution[0].cluster_classifier.dbscan_range
                print solution[1]

                if solution[1] > best_solution[1]:
                    best_solution=solution
            p.join()


        casc.transformers[e]= best_solution[0]
        graphs =[[best_solution[0].transform(graphs[a][b]) for b in range(2)] for a in range(4) ]
        #draw.graphlearn(graphs[0][0][:5])
        print "LEVEL DONE"
    return




def test_repeats(trans, graphs, repeats=3, graph_num=600):
    try:
        SIZEHALF = graph_num/2
        # need to pick parameters to test...
        trans.score_threshold, trans.cluster_classifier.dbscan_range =  random.uniform(0.03,0.3), random.uniform(0.35,0.80)
        trans.prepfit()
        # train and get acc REPEAT times...
        accs=[]
        for ex in map(lambda x:copy.deepcopy(trans), range(repeats)):
            #ex.prepfit()
            ex.annotator.fit( *(util.sample_pos_neg(graphs[0][0] ,graphs[0][1],SIZEHALF,SIZEHALF)[0]))
            # annotate the second graphset:
            g,h= util.sample_pos_neg(graphs[1][0] , graphs[1][1],SIZEHALF,SIZEHALF)[0]
            graphss = ex.annotator.transform(  g+h )
            accs.append(test_once(ex, graphss, graphs))


        return (ex,np.array(accs).mean()-np.array(accs).var())
    except:
        return (trans,0)

def test_once(ex, graphss, graphs):

    # get subgraphs from second set  and fit classifier on clusters
    subgraphs = list(ex.abstractor.get_subgraphs(  graphss) )
    ex.cluster_classifier.fit(subgraphs)
    ex.abstractor.nameestimator = ex.cluster_classifier

    # transform 3rd and  4th set  -> train and test ;; get accuracy
    re = [ex.transform(graphs[a][b]) for a,b in [(2,0),(2,1),(3,0),(3,1)]]
    acc= util.graphs_to_acc(*re)
    return acc




