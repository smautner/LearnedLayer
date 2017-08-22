import random
import pprint
from moleLearnedLayer import util as util
from copy import deepcopy
from eden.graph import vectorize




def get_default_samplers_params():
    grammar_options={"radius_list":list(range(random.randint(1,4))),
            "thickness_list":list(range(1,random.randint(2,4))),
            "min_cip_count":random.randint(1,3),
            "min_interface_count":random.randint(2,3),
            }


    sampler_options={
        "grammar_options":grammar_options,

        random.choice(["core_choice_byfrequency",
                       "core_choice_byscore",
                       "core_choice_bytrial",
                       "size_constrained_core_choice"]):True,
        "n_steps":random.randint(5,50),
        "quick_skip_orig_cip":False#random.choice([True,False]),
    }
    if "core_choice_bytrial" in sampler_options:
        sampler_options["core_choice_bytrial_multiplier"]=random.random()+.5
    if "size_constrained_core_choice" in sampler_options:
        sampler_options['size_constrained_core_choice'] = random.randint(0,8)




    if random.random()>0.5:
        a=random.random()
        b=random.random()
        if a > b:
            a,b=b,a
        sampler_options['improving_threshold_fraction'] = b
        sampler_options['improving_linear_start_fraction'] = a

    if random.random()>0.5:
        '''tried setting random value for max and min -> crashes always.. '''
        a=random.random()
        if random.random()> .5:
            sampler_options['orig_cip_max_positives'] = a
        else:
            sampler_options['orig_cip_min_positives'] = a

    if random.random()>.5:
        sampler_options['accept_static_penalty']=random.random()

    return sampler_options


def get_learned_samplers_params():

    '''
    depth=2,
    max_group_size=6,
    min_group_size=2,
    group_score_threshold=0,
    '''
    params=get_default_samplers_params()

    if random.random()>.5:
        a=random.sample(range(2,11),2)
    else:
        a=[2,6]
    learn_params={
        "depth": random.randint(0,6),
        "max_group_size":max(a),
        "min_group_size":min(a),
        "group_score_threshold": random.random() # WAT?
    }

    params['learn_params'] = learn_params

    return params

def get_random_params(type):
    if type==0:
        return get_default_samplers_params()
    if type==1:
        return get_learned_samplers_params()
    if type==2:
        # return get_cycle_samplers_params(), cycle has no params on its own
        return get_default_samplers_params()



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
    return 'optimizer_run_%d_%d' % (typ,run_id)

def run(data,typ,run_id):
    params=get_random_params(typ)
    sampler = make_sampler(params,typ)
    results=[]
    for gpos,gneg in util.loadfile(data):
        #task = namedtuple("task",'samplerid size repeat sampler neg pos')
        results.append(  util.sample( util.task( typ, len(gpos),0,  deepcopy(sampler),gneg,gpos)) )
    util.dumpfile( (results,params) , get_optout_fname(typ,run_id))



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


def report(aid,typ,numtries, top=5, show=False):
    esti = util.aid_to_linmodel(aid)
    data= [ util.loadfile(get_optout_fname(typ,e+1)) for e in range(numtries) ]


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





