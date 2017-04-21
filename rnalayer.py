import os.path
import random
from eden_rna.io.fasta import load
import itertools
from eden.util import selection_iterator

import urllib


def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)

def rfam_uril(family_id):
    return '%s.fa'%(family_id)

def rfamhandler(famid):
    filename=rfam_uril(famid)
    if not os.path.isfile(filename):
        urllib.urlretrieve(rfam_uri(famid),filename)
    return filename
    

def get_sequences(size=9999,withoutnames=False):
    sequences = itertools.islice( load(rfamhandler("RF00005")), size)
    if withoutnames:
        return [ b for (a,b) in sequences ]
    return sequences


def get_data2(rfamid,repeats, trainsizes=[10,20]):
    allseqs = list(get_sequences() )
    return [[getseqs(allseqs,amount) for amount in trainsizes] 
            for repeat in range(repeats)]


def getseqs(allsequences, amount):
    random.shuffle(allsequences)
    return allsequences[:amount]
    












# STUFF BELOW SHOULD BE CRAP 

def get_data(rfam_id_1,rfam_id_2):
    '''

    Parameters
    ----------
    rfam_id

    Returns
    -------
        vectors, classes, graphz(tabun sequences,ne), stats
    '''


    #   rfam_id to
    #active_X = pipe(assay_id, download_active, babel_load, vectorize)
    #inactive_X = pipe(assay_id, download_inactive, babel_load, vectorize)


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
