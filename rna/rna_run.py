import matplotlib
matplotlib.use('Agg')
import numpy as np
from layerutils import make_inbetween_plot
import pprint
from rna_getdata import get_data2
from rna_getsamplers import make_samplers_rna

import time
from layerutils import transpose
from graphlearn01.minor.rna.infernal import infernal_checker

def runner(sampler,problem):
    print 'runner start,'
    starttime = time.time() 
    thing= list(sampler.fit_transform(problem))
    # if sampling stopps at step 0, e[0] does not exist, we filter these with len e > 0
    result = [ e[0] for e in thing if len(e)>0]
    print ".. done"
    return (result, time.time()-starttime)









def run_experiments(data,samplers):
    
    #0  is default
    #1  one is the forgi
    #2  two is the learned guy 
    #samplers=[samplers[2]]

    return [[[ runner(sampler,problem)
        for problem in repeat]
            for repeat in  data] 
                for sampler in samplers]




def evaluate(rawrun):

    means=[]
    stds=[]
    means_time=[]
    stds_time=[]

    for sampler_data in rawrun: 

       temp_m=[]
       temp_s=[]
       temp_m_time=[]
       temp_s_time=[]
       data=transpose(sampler_data) 
       for runsofsamelengh in data: 
           seq,tim = transpose(runsofsamelengh)
           #print seq,tim
           seq = [result[1] for run in seq for result in run]
           seq = infernal_checker(seq,cmfile='../tools/rf00005.cm', cmsearchbinarypath='../tools/cmsearch')
           temp_m.append(np.mean(seq))
           temp_s.append(np.std(seq))
           temp_m_time.append(np.mean(tim))
           temp_s_time.append(np.mean(tim))
       means.append(temp_m)
       stds.append(temp_s)
       means_time.append(temp_m_time)
       stds_time.append(temp_s_time)
    return means,stds, means_time, stds_time


trainsizes=[10,20,50,100,200,300,400]
if False:    
    data= get_data2('RF00005',repeats=3, trainsizes=trainsizes)
    samplers = make_samplers_rna(n_jobs=3)
    rawrun = run_experiments(data,samplers)
    print "ran the experiments"
    means,stds,means_time, stds_time = evaluate(rawrun)
    
    print "got data"
    print means,stds,means_time,stds_time

    make_inbetween_plot(labels=trainsizes, means=means , stds=stds, fname='rna.png')
    make_inbetween_plot(labels=trainsizes, means=means_time, stds=stds_time,fname='rna_time.png',dynamic_ylim=True)


#################################
#  debug run :)
###############################

trainsizes=[400]
if False:
    data= get_data2('RF00005',repeats=1, trainsizes=trainsizes)
    samplers = make_samplers_rna(n_jobs=3)
    samplers=[samplers[2]]
    rawrun = run_experiments(data,samplers)
    print "ran the experiments"
    means,stds,means_time, stds_time = evaluate(rawrun)
    print means,stds,means_time,stds_time
    make_inbetween_plot(labels=trainsizes, means=means , stds=stds, fname='rna.png')
    make_inbetween_plot(labels=trainsizes, means=means_time, stds=stds_time,fname='rna_time.png',dynamic_ylim=True)

