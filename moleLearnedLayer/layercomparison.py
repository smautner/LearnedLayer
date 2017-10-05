
import os

import matplotlib.pyplot as plt
import numpy as np

import toolz

import util
########
# WRITE TASK FILE AND RUN
##########
from moleLearnedLayer.util import graphs_to_scores


def make_task_file(aid='1834',sizes=[50,75,100],repeats=2,params=[{},{},{}], selectsamplers=[0,1,2], taskfile_poststring=''):
    '''
    we drop lots of these in 1 file:
    task = namedtuple("task",'samplerid size repeat sampler neg pos')
    '''
    pos,neg = util.getgraphs(aid)
    tasks=[]
    models=[]
    for size in sizes:
        repeatsXposnegsamples,estis = util.sample_pos_neg_ESTI(pos,neg,size,size,repeats)
        models.append(estis)
        for i, sampler in enumerate(util.get_all_samplers(params=params, select=selectsamplers)):
            for j, (pos_sample, neg_sample) in enumerate(repeatsXposnegsamples):
                tasks.append(util.task(i,size,j,sampler,neg_sample,pos_sample))



    fname = "%s/%d%s" % (aid,max(sizes), taskfile_poststring)
    util.dumpfile(tasks,fname)
    util.dumpfile(models,fname+"_models" )
    return fname

def showtask(filename, taskid):
    tasks= util.loadfile(filename)
    print tasks[taskid]

def run(filename, taskid):
    tasks= util.loadfile(filename)
    try:
        result = util.sample(tasks[taskid])
    except Exception as exc:
        print "molelearnedlayer except"
        #print tasks[taskid]
        #import traceback
        #print traceback.format_exc(20)
        return None
    util.dumpfile(result,getresfilename(filename,taskid))


###################
#n EVALUATE
##################
def getresfilename(filename,taskid):
    return "res_%s_%d" % (filename,taskid)

def readresults(filename,taskcount):
    #return [util.loadfile( "res_%s_%d"  %(filename,i)) for i in range(taskcount) ]
    data=[]
    for e in range(taskcount):
        path= getresfilename(filename,e)
        if os.path.exists(path):
            data.append(util.loadfile(path))
    return data

def getcol(procress):
    coldict={1:'#F94D4D',
         2:'#555555',
         0:'#6A9AE2'}
    return coldict[procress.samplerid]


def _get_odict(oracles,sizes):
    odict={}
    for oracle_allrepeats,size in zip(oracles,sizes):
        odict[size]={i:esti for i,esti in enumerate(oracle_allrepeats)  }
    return odict



def eval(res,oracles,sizes):
    processed=[]

    odict=_get_odict(oracles,sizes)

    for byrepeats in toolz.groupby(lambda x:x.samplerid+x.size, res).values():

        time=np.array([ x.time for x in byrepeats ])

        scores_by_rep =  [ util.graphs_to_scores(x.graphs,odict[x.size][x.repeat])  for x in byrepeats ]
        scores= np.concatenate( scores_by_rep )

        processed.append( util.processed_result(byrepeats[0].samplerid,byrepeats[0].size,scores.mean(),scores.var(),time.mean(),time.var(), [e.mean() for e in scores_by_rep])  )


    return processed



def draw(processed,filename, get_mean = lambda x: x.score_mean , get_var=lambda x:x.score_var,show=False): # see runner :)
    # this is shitty and boring, but it works atm..
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

    plt.legend(loc=4)
    plt.savefig(filename)
    if show:
        plt.show()


def draw2(processed,filename, get_mean = lambda x: x.score_mean , get_var=lambda x:x.score_var,show=False):
    # the plan is to remove the shades, add dots and regress the curve..  also needs logistic regression and a legend
    plt.figure(figsize=(15,5))


    # loop over curves...
    for oneline in toolz.groupby(lambda x:x.samplerid, processed).values():

        # X,y values for fitting curve
        oneline.sort(key=lambda x:x.size)
        sizes = [x.size for x in oneline]
        y_values = np.array([get_mean(x) for x in oneline])

        #func = xy_to_curve(sizes,y_values)

        # color and legend
        legend= samplerid_to_samplername(oneline[0].samplerid)
        col = getcol(oneline[0])


        # draw curve
        #show_x=np.array(range(0,1000))
        #plt.plot(show_x, [func(show) for show in show_x], label=legend, color=col) #same as line above \/
        plt.plot(sizes, y_values, label=legend, color=col) #same as line above \/



        # also draw the subresults

        def getsubres():
            for procress in oneline:
                for repeatscore in procress.score_sub_means:
                    yield(procress.size,repeatscore)

        subres= zip(*list(getsubres()))
        plt.plot(subres[0], subres[1],'o', color=col)



    plt.legend(loc='lower right')
    plt.savefig(filename)
    if show:
        plt.show()



def xy_to_curve(X,y):
    '''
    # functions
    def func(x, a, b,c):
        return a*((x-c)**b)
    '''
    def func(x,a,b,c):
        return(  a+b/(x**c)  )
    # fit
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, X, y)
    return  lambda x:func(x,*popt)






def samplerid_to_samplername(i):
    # see clean make  util -> make samplers chem if the order is awrite:)
    return {0:'Default',1:'Learned',2:"Cycles"}[i]


def evalandshow(fname,tasknum,sizes,show=False):
    #oracle = util.aid_to_linmodel(aid)

    res = readresults(fname,tasknum)
    processed = eval(res,util.loadfile(fname+"_models"),sizes)

    util.dumpfile((processed,fname,show),"ASASD")
    draw2(processed,fname+"score.png", show=show)
    draw(processed,fname+"time.png",get_mean=lambda x:x.time_mean,get_var=lambda x:x.time_var, show=show)




if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2])-1)







