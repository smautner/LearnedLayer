import random
import os
import pprint
from moleLearnedLayer import util as util
from copy import deepcopy
from eden.graph import vectorize




def get_default_samplers_params():

    grammar_options={"radius_list":[0,1],
            "thickness_list":[2],
            "min_cip_count": 2,
            "min_interface_count": 2,
            }
    sampler_options={
        "grammar_options":grammar_options,
        "core_choice_byfrequency":True,
        "n_steps":25,
        "quick_skip_orig_cip":True
    }
    sampler_options['accept_static_penalty']=.2
    return sampler_options


def getallparams():
    allparams=[]
    for dilude in [False]: # true seems bad
        for treshpercent in range(10,26):
            tresh = treshpercent/100.0
            for depth in [2,3]:
                params=get_default_samplers_params()
                learn_params={
                'dbscan_range': .75,
                'annotate_dilude_score': dilude,
                'depth': depth,
                'group_score_threshold': tresh,
                'min_clustersize': 5,  # this is an exclusive parameter :)
                'max_group_size': 7,
                'min_group_size': 2}
                params['learn_params'] = learn_params
                allparams.append(params)
    return allparams



def get_params(runid):
    params=getallparams()
    return params[runid]



def make_sampler(params,typ):
    if typ == 0:
        return util.get_no_abstr(kwargs=params)
    if typ == 1:
        return util.get_casc_abstr(kwargs=params)
    if typ == 2:
        return util.get_hand_abstr(kwargs=params)







####
# actual interface :)
####
def get_optout_fname(typ,run_id):
    return 'optout_grid/optimizer_run_%d_%d' % (typ,run_id)

def run(data,typ,run_id):
    fname = get_optout_fname(typ,run_id)
    if os.path.exists(fname):
        print 'not rerunning %d' % run_id
        return

    params=get_params(run_id-1)

    sampler = make_sampler(params,typ)
    results=[]
    for gpos,gneg in util.loadfile(data):
        #task = namedtuple("task",'samplerid size repeat sampler neg pos')
        results.append(  util.sample( util.task( typ, len(gpos),0,  deepcopy(sampler),gneg,gpos)) )
    util.dumpfile( (results,params) , fname)



def run_many(data,typ=1,num_tries=20):
    """
    :param data:        see util init_optimisation
    :param typ:         sampler type to run on
    :param num_tries:   number of evals that are conducted
    :return:
        dumps ([utils.sampled for e in data],paramsdict) > "optimizer_run_TYP_NUMTRY"
    """
    for i in range(num_tries):
        run(data,typ,i)


if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2]), int(sys.argv[3])  )


#####
#
#####
from collections import defaultdict
def merge_dicts(l):
    res=defaultdict(list)
    for d in l:
        for k,v in d.items():
            res[k].append(v)
    for k,v in res.items():
        if type(v[0])==dict:
            res[k]=merge_dicts(v)
    return dict(res)




def check_output(typ,numtries):
    import os
    for e in range(numtries):
        if not os.path.exists(get_optout_fname(typ,e+1)):
            print e+1

def report(aid,typ,numtries, top=5, show=False):
    esti = util.aid_to_linmodel(aid)
    import os

    data=[]
    for e in range(numtries):
        path= get_optout_fname(typ,e+1)
        if os.path.exists(path):
            data.append(util.loadfile(path))


    #data= [ util.loadfile(get_optout_fname(typ,e+1)) for e in range(numtries) ]


    def cleandata(data):
        for samplist,params in data:
            graphs = [g for e in samplist for g in e.graphs]
            if len(graphs)> 0:
                yield (esti.decision_function(vectorize(graphs,n_jobs=1)).mean() , params)

    results= [a for a in cleandata(data)]

    print "%d of %d experiments crashed" % ( numtries-len(results),  numtries)

    results.sort(key=lambda x:x[0],reverse=True)

    if show:
        import matplotlib.pylab as plt
        plt.figure(figsize=(12, 6))
        plt.hist([x[0] for x in results], 20, normed=1, alpha=.8, histtype='step', stacked=False, fill=True)
        plt.show()

    getparms = lambda x: merge_dicts( [ params for score,params in x ] )
    best,worst = getparms( results[:top] ), getparms (results[-top:])
    pprint.pprint(best)
    pprint.pprint(worst)





