


import util
from graphlearn01.learnedlayer.cascade import Cascade
import copy
import random




def get_graph_stack(aid="651610", overeight_graph_num=600):
    a,b=util.getgraphs(aid)
    return util.sample_pos_neg_no_duplicates(a,b,overeight_graph_num,4)


def optimize(graphs, depth,graph_num,num_tries_per_depth_level=30):
    casc = Cascade(depth=depth,debug=False,max_group_size=20)
    casc.setup_transformers()
    SIZEHALF = graph_num/2

    # optimize each level of the cascade
    for e in range(depth):

        best_solution=(None,-1)

        # use the first graphset to train an estimator:
        trans = casc.transformers[e]
        trans.prepfit()
        trans.annotator.fit(   *(util.sample_pos_neg(graphs[0][0] ,graphs[0][1],SIZEHALF,SIZEHALF)[0]))

        # annotate the second graphset:
        g,h= util.sample_pos_neg(graphs[1][0] , graphs[1][1],SIZEHALF,SIZEHALF)[0]
        graphss = trans.annotator.transform(  g+h )


        for ex in map(lambda x:copy.deepcopy(trans), range(num_tries_per_depth_level)):
            try:

                #def test_values(ex,graphss,graphs):

                # set value we want to try and init
                ex.score_threshold, ex.cluster_classifier.dbscan_range =  random.uniform(0.05,0.55), random.uniform(0.55,0.95)
                ex.prepfit()
                ex.annotator= trans.annotator

                # get subgraphs from second set  and fit classifier on clusters
                subgraphs = list(ex.abstractor.get_subgraphs(  graphss) )
                print "there are %d subgraphs extracted from %d graphs" % (len(subgraphs),len(graphss)),ex.score_threshold, ex.cluster_classifier.dbscan_range
                ex.cluster_classifier.fit(subgraphs)
                ex.abstractor.nameestimator = ex.cluster_classifier

                # transform 3rd and  4th set  -> train and test ;; get accuracy
                re = [ex.transform(graphs[a][b]) for a,b in [(2,0),(2,1),(3,0),(3,1)]]
                acc= util.graphs_to_acc(*re)


                if acc>best_solution[1]:
                    best_solution=(ex,acc)
                print acc
            except:
                continue

        casc.transformers[e]= best_solution[0]


        # transform all the sets with the best transformer to get the input for the next round
        #[[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)]]
        #[[winrar[0].transform(graphs[a][b]) for b in range(2)] for a in range(4) ]

        graphs =[[best_solution[0].transform(graphs[a][b]) for b in range(2)] for a in range(4) ]
        print "LEVEL DONE"
    return