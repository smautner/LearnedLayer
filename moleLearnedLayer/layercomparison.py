

import matplotlib.pyplot as plt
import numpy as np

import toolz

import util
########
# WRITE TASK FILE AND RUN
##########
from moleLearnedLayer.util import graphs_to_scores


def make_task_file(aid='1834',sizes=[50,75,100],repeats=2,params=[{},{},{}], selectsamplers=[0,1,2]):
    '''
    we drop lots of these in 1 file:
    task = namedtuple("task",'samplerid size repeat sampler neg pos')
    '''
    pos,neg = util.getgraphs(aid)
    tasks=[]
    for size in sizes:
        repeatsXposnegsamples = util.sample_pos_neg(pos,neg,size,size,repeats)
        for i, sampler in enumerate(util.get_all_samplers(params=params, select=selectsamplers)):
            for j, (pos_sample, neg_sample) in enumerate(repeatsXposnegsamples):
                tasks.append(util.task(i,size,j,sampler,neg_sample,pos_sample))

    util.dumpfile(tasks,"%s_%d" % (aid,max(sizes)))
    return "%s_%d" % (aid,max(sizes))

def showtask(filename, taskid):
    tasks= util.loadfile(filename)
    print tasks[taskid]

def run(filename, taskid):
    tasks= util.loadfile(filename)
    try:
        result = util.sample(tasks[taskid])
    except Exception as exc:
        print "molelearnedlayer is showing the task object:"
        print tasks[taskid]
        import traceback
        print traceback.format_exc(20)
        return None
    util.dumpfile(result,"res_%s_%d" % (filename,taskid))


###################
#n EVALUATE
##################

def readresults(filename,taskcount):
    return [util.loadfile( "res_%s_%d"  %(filename,i)) for i in range(taskcount) ]


def getcol(procress):
    coldict={1:'#F94D4D',
         2:'#555555',
         0:'#6A9AE2'}
    return coldict[procress.samplerid]


def eval(res,oracle):
    processed=[]
    for byrepeats in toolz.groupby(lambda x:x.samplerid+x.size, res).values():
        time=np.array([ x.time for x in byrepeats ])
        graphs = [g for x in byrepeats for g in x.graphs ]
        scores = graphs_to_scores(graphs, oracle)
        processed.append( util.processed_result(byrepeats[0].samplerid,byrepeats[0].size,scores.mean(),scores.var(),time.mean(),time.var())  )
    return processed

def draw(processed,filename, get_mean = lambda x: x.score_mean , get_var=lambda x:x.score_var,show=False): # see runner :)
    plt.figure(figsize=(15,5))

    for oneline in toolz.groupby(lambda x:x.samplerid, processed).values():
        oneline.sort(key=lambda x:x.size)
        sizes = [x.size for x in oneline]
        y_values = np.array([get_mean(x) for x in oneline])
        y_variances= np.array([get_var(x) for x in oneline])
        col = getcol(oneline[0])
        plt.fill_between(sizes, y_values+y_variances , y_values -y_variances, facecolor=col,
                         alpha=0.15, linewidth=0,label='%s' % samplerid_to_samplername(oneline[0].samplerid))
        plt.plot(sizes,y_values,color=col)

    plt.legend()
    plt.savefig(filename)
    if show:
        plt.show()


def samplerid_to_samplername(i):
    # see clean make  util -> make samplers chem if the order is awrite:)
    return {0:'noabstr',1:'learned',2:"hand"}[i]


def evalandshow(aid,fname,tasknum,show=False):
    oracle = util.aid_to_linmodel(aid)
    res = readresults(fname,tasknum)
    processed = eval(res,oracle)
    draw(processed,fname+"score.png", show=show)
    draw(processed,fname+"time.png",get_mean=lambda x:x.time_mean,get_var=lambda x:x.time_var, show=show)




if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2])-1)







