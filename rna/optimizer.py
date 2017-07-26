
import rna_getdata as data
import rna_eval as eva
import rna_getsamplers as getsamps
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
        "n_steps":random.randint(5,30),
        "quick_skip_orig_cip":random.choice([True,False]),
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
        a=random.random()
        b=random.random()
        if a > b:
            a,b=b,a
        sampler_options['orig_cip_max_positives'] = b
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

    if random.randint()>.5:
        a=random.randint()
        b=random.randint()
        if a > b:
            a,b=b,a
    else:
        a=2
        b=6

    learn_params={
        "depth": random.randint(2,4),
        "max_group_size": b,
        "min_group_size":a,
        "group_score_threshold": random.randint/.7
    }

    params['learn_params'] = learn_params

    return params


def get_random_params(type):
    if type==0:
        return get_default_samplers_params()
    if type==1:
        return get_learned_samplers_params()
    #if type=2:
        #return get_forgi_samplers_params()



def make_sampler(params,typ):
    if typ == 0:
        return getsamps.get_default_sampler(kwargs=params)
    if typ == 1:
        return getsamps.get_learned_sampler(kwargs=params)


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


def run_once(scale=50, samplertype_int=None):
    # get data
    all_seqs=list(data.get_sequences(rfamid="RF00005"))
    a=data.getseqs(all_seqs,scale)
    b=data.getseqs(all_seqs,scale)

    # get random parameters and sampler
    params = get_random_params(samplertype_int)

    print "run_once params" ,params
    # run
    resa=run_sampler_wrap(make_sampler(params,samplertype_int),a)
    resb=run_sampler_wrap(make_sampler(params,samplertype_int),b)
    return "%.4f__%.4f__%s\n" % (resa,resb,params)




# ok so we get an array id to know where to write the results


types_of_samplers=2
num_sequences=50

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:    # NO ARGS
        print "need to know where to write my results"
        exit()

    jobno=int(sys.argv[1])
    jobtype={0:"default",1:"learned",2:"hand"}

    while True:
        res = run_once(scale=num_sequences, samplertype_int=jobno % 3)
        with open("res_%s_%d" % (jobtype[jobno%types_of_samplers],jobno), "a") as myfile:
            myfile.write(res)


"""
TODO:
1. dump errors somewehre else..
2. duplicate this for the chem case

"""