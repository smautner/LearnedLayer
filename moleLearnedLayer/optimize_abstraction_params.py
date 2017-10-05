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


def getparams():
    return {     'dbscan_range': random.uniform(.5,.7),
                        'depth': random.randint(2,4),
                        'group_score_threshold': random.randint(3,17)/100.0,
                        'min_clustersize': random.randint(10,100)/1000,  # this is an exclusive parameter :)
                        'max_group_size': random.randint(6,10),
                        'min_group_size': random.randint(2,4)}

def getacc( pos,neg, post,negt ):
    esti= util.graphs_to_linmodel(pos,neg)
    X,y = util.graphs_to_Xy(post,negt)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)
    return acc


def getsampler(param):
    paramz=optimize.get_default_samplers_params()
    paramz['learn_params']=param
    return util.get_casc_abstr(kwargs=paramz)

def run(filename, taskid):
    tasks = util.loadfile(filename)
    param = getparams()



    scores=[]
    for a in tasks:
        sampler=getsampler(param)
        pos,neg, testpos,testneg =  a
        alltrain = [sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in sampler.graph_transformer.fit_transform(pos,neg,
                        remove_intermediary_layers=False)]
        alltest= [sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in sampler.graph_transformer.transform(testpos+testneg,remove_intermediary_layers=False)]
        s= len(pos)
        ss=len(testpos)
        scores.append(  getacc( alltrain[:s],alltrain[s:],alltest[:ss],alltest[ss:]))

    util.dumpfile((np.median(scores),param),"oap/%s_%d" % (filename,taskid))

def eva(filename, tasknum):
    stuffs = [ util.loadfile("oap/%s_%d" % (filename,i)) for i in range(tasknum)]

    stuffs.sort(reverse=True)
    pprint.pprint(stuffs[0][1])


if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2])-1)

# BELOW SHOULD BE JUST CRAP
############################################
#######################################################3

def get_graph_stack(aid="651610", overeight_graph_num=600):
    a,b=util.getgraphs(aid)
    return util.sample_pos_neg_no_duplicates(a,b,overeight_graph_num,4)


def optimize(graphs, depth,graph_num,num_tries_per_depth_level=30):
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




