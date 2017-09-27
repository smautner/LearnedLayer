import numpy as np
import util
from graphlearn01.learnedlayer.cascade import Cascade
import copy
import random
from multiprocessing import Pool
import eden
from graphlearn01.utils import draw


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
            p=Pool(20)
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
        trans.score_threshold, trans.cluster_classifier.dbscan_range =  random.uniform(0.05,0.55), random.uniform(0.55,0.95)
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


        return (ex,np.array(accs).mean())
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

