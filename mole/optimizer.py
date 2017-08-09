import clean_sample
from eden.util import selection_iterator
import numpy as np
import clean_eval as eva
import clean_make_tasks as sampsNdata
import random

import logging
from eden.util import configure_logging
configure_logging(logging.getLogger(),verbosity=3)

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
        "n_steps":random.randint(5,10),                                                     # !!! only 10 is max here
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
        "depth": random.randint(2,4),
        "max_group_size":max(a),
        "min_group_size":min(a),
        "group_score_threshold": random.random()/.7 # WAT?
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
        return sampsNdata.get_no_abstr(kwargs=params)
    if typ == 1:
        return sampsNdata.get_casc_abstr(kwargs=params)
    if typ == 2:
        return sampsNdata.get_hand_abstr(kwargs=params)


'''
def avgscore(sequences):
    res=eva.sequences_to_scores(sequences,"RF00005")
    #print "avgscore", res
    return res.mean()

def run_sampler_wrap(sampler,data):
    thing= list(sampler.fit_transform(data))
    print "run_sampler thing", thing
    result = [ e[0][1] for e in thing if len(e)>0]
    print "run_sampler result", result
    return avgscore(result)
'''
def run_once(alldata, scale=50, samplertype_int=None, forceparams=None):
    # get data
    X,y,graphs_p,graphs_n,esti = alldata
    poslist = np.random.permutation(range(len(graphs_p)))[:scale]
    neglist = np.random.permutation(range(len(graphs_n)))[:scale]
    neg= list(selection_iterator(graphs_n, neglist))
    pos= list(selection_iterator(graphs_p, poslist))

    # get random parameters and sampler
    if forceparams:
        params=forceparams
    else:
        params = get_random_params(samplertype_int)
    taski=sampsNdata.task(samplertype_int,scale,0,make_sampler(params,samplertype_int),neg,pos)

    print "run_once params" ,params
    if True:# debug
        with open("params","w") as f:
            f.write(str(params))
    # run
    run=clean_sample.runwrap(taski)
    resa=eva.graphs_to_scores(run.graphs,esti)
    return "%.4f__%s\n" % (resa.mean(),params)




# ok so we get an array id to know where to write the results

numgraphs=150
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:    # NO ARGS
        print "need to know where to write my results"
        exit()

    alldata= sampsNdata.get_data("1834")

    if sys.argv[1]=="debug_last":
        with open("params","r") as f:
            params=eval(f.read())
        print run_once(alldata,samplertype_int=1,forceparams=params)
        exit()


    jobno=int(sys.argv[1])
    jobtype={0:"default",1:"learned",2:"hand"}

    while True:
        res = run_once(alldata,scale=numgraphs, samplertype_int=(jobno % 3))
        with open("res_%s_%d" % (jobtype[jobno%3],jobno), "a") as myfile:
            myfile.write(res)


"""
TODO:
    make samplers needs to accept the dict. 
    
"""
