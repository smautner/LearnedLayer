import toolz
import util
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from collections import Counter
from scipy.optimize import curve_fit


import layercomparison as lc

#########
# histogram
####

def loadgraphs(fname):
    tasks = util.loadfile(fname)
    # we now have a bunch of tasks
    #task = namedtuple("task",'samplerid size repeat sampler neg pos')
    res=[]
    for i, task in enumerate(tasks):
        if task.samplerid == 1: # see util if this is still correct :)
            sampl = util.loadfile("res_%s_%d" % (fname,i))
            # sampled = namedtuple("sampled",'samplerid,size,repeat,time, graphs')
            res.append( [task.pos, sampl.graphs, task.size, task.repeat] )

    res.sort(key=lambda x: x[2])
    return res

def loadgraphs(fname,taskcount):
    lc.readresults(fname, taskcount)




def histo(fname,sizes,taskcount):

    oracles = util.loadfile(fname+"_models")
    odict = lc._get_odict(oracles,sizes)

    sampled = lc.readresults(fname,taskcount)
    tasks= util.loadfile(fname)

    hs = lambda s: s.samplerid+s.size+s.repeat*10000
    taskindex = { hs(t):t  for t in tasks  }


    for sampld in sampled:
        print "size: %d   repeat: %d" % (sampld.size,sampld.repeat)
        seed = util.graphs_to_scores( taskindex[hs(sampld)].pos , odict[sampld.size][sampld.repeat] )
        gen = util.graphs_to_scores(  sampld.graphs,  odict[sampld.size][sampld.repeat] )

        plt.figure(figsize=(12,6))
        plt.hist((seed,gen), 20, normed=1, alpha=.8,histtype='step', stacked=False, fill=True,label=['seed','generated'])
        plt.legend()
        plt.show()


def comparisondata_to_histo(fname,aid, count):
    stuff = loadgraphs(fname,count)
    histo(stuff,aid)




###############
# learning curve things
###############

def learning_curve_wrap(aid, train_sizes):
    p,n = util.getgraphs(aid)
    size= min(len(p),len(n))
    p,n = util.sample_pos_neg(p,n,size,size,repeats=1)[0]
    X,y = util.graphs_to_Xy(p,n)
    X,y = shuffle(X,y)


    train_sizes, train_scores_crap, valid_scores = learning_curve(
        SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=8, loss='log'),
        X,
        y,
        train_sizes=train_sizes,
        cv=5)

    return train_sizes, train_scores_crap, valid_scores


def learning_curve_to_xy(train_sizes, valid_scores):
    x = np.array(train_sizes, dtype=float)
    y = [ row.mean() for row in  (valid_scores)]
    return x,y



def xy_to_curve(X,y):

    '''
    # functions
    def func(x, a, b,c):
        return a*((x-c)**b)

    def inverse_func(y,popt):
        a,b,c = popt
        return np.power(y/a,1.0/b)+c

    '''
    def func(x,a,b,c):
        return(  a+b/(x**c)  )

    def inverse_func(y,popt):
        a,b,c = popt
        return np.power(b/(y-a),1.0/c)

    # fit
    popt, pcov = curve_fit(func, X, y, method='dogbox')
    #print popt , "1st element is a, second is b -> now its easy to find the x to a y "



    # plot
    plt.plot(X, y, 'bo',label="Original Data")
    plt.ylim((min(y)-.05, max(y)+.05))
    show_x=np.array(range(0,1000))
    plt.plot(show_x, [func(show  , *popt) for show in show_x], label="Fitted Curve") #same as line above \/
    plt.legend(loc='lower right')
    plt.show()

    return lambda x: inverse_func(x,popt), lambda x:func(x,*popt)


def crossval_to_inversefunc(assay_id,train_sizes):
    _,_,valid_scores= learning_curve_wrap(assay_id,train_sizes)
    x,y=learning_curve_to_xy(train_sizes,valid_scores)
    return xy_to_curve(x,y)


def lastplot(func,x,y):
    print x,y

    #plt.ylim((min(y)-.05, max(y)+.05))
    plt.ylim(.2, .8)


    # original line
    show_x=np.array(range(0,1000))
    plt.plot(show_x, [func(xx) for xx in show_x], label="normal trining curve") #same as line above \/

    # our new stuff
    plt.plot(x,y,'ro',label='training curce with with generated instances')

    plt.legend(loc='lower right')

    plt.show()
############
# running stuff
############




#
def make_task_file2(aid='1834',sizes=[50,75,100],test=50,repeats=2,params=[{},{},{}], selectsamplers=[0,1,2]):
    '''
    we drop lots of these in 1 file:
    task = namedtuple("task",'samplerid size repeat sampler neg pos')
    '''
    pos,neg = util.getgraphs(aid)

    tasks=[]
    for size in sizes:
        repeatsXposnegsamples = util.sample_pos_neg(pos,neg,size+test,size+test,repeats)
        repeatsXposnegsamples = [ [pos2[:size],neg2[:size],pos2[size:], neg2[size:] ]  for pos2,neg2 in repeatsXposnegsamples]
        for i, sampler in enumerate(util.get_all_samplers(params=params, select=selectsamplers)):
            for j, (pos_sample, neg_sample ,pos_test,neg_test) in enumerate(repeatsXposnegsamples):
                tasks.append(util.task2(i,size,j,sampler,neg_sample,pos_sample,neg_test,pos_test))

    util.dumpfile(tasks,"natlearn/t2_%s_%d" % (aid,max(sizes)))
    return "t2_%s_%d" % (aid,max(sizes))



def gettask(tasks,idd):
    task = tasks[idd%len(tasks)]
    if idd < len(tasks):
        return task
    else:
        # task = namedtuple("task",'samplerid size repeat sampler neg pos') # reverse pos/ neg
        return util.task( task.samplerid,task.size,task.repeat,task.sampler,task.pos,task.neg)

def run(filename, taskid):
    tasks = util.loadfile("natlearn/"+filename)
    task=gettask(tasks,taskid)
    try:
        result = util.sample(task)
    except Exception as exc:
        print "naturallearning is showing the task object:"
        print task
        import traceback
        print traceback.format_exc(20)
        return None

    util.dumpfile(result,"natlearn/res_%s_%d" % (filename,taskid))


if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2])-1)


def eva(aid,fname,tasknum):
    '''
    repeat and size should id a task thing... that should give the test cases and train graphs
    then we load the results to get the gen graphs.
    '''

    tasks = util.loadfile(fname)
    tasks = toolz.groupby( lambda x:x.size, tasks)
    respos= toolz.groupby( lambda x : x.size, [ util.loadfile( "res_%s_%d" % (fname,i)) for i in range(tasknum) ]  )
    resneg= toolz.groupby( lambda x : x.size, [ util.loadfile( "res_%s_%d" % (fname,i)) for i in range(tasknum,tasknum*2) ]  )

    RESULT=[]
    for k in tasks: # the same k is also in respos and resneg

        tasks_byrepeat= toolz.groupby(lambda x:x.repeat, tasks[k])
        respos_byrepeat= toolz.groupby(lambda x:x.repeat, respos[k])
        resneg_byrepeat= toolz.groupby(lambda x:x.repeat, resneg[k])

        means=[]
        for j in tasks_byrepeat:
            task=tasks_byrepeat[j][0]
            pos=respos_byrepeat[j][0]
            neg=resneg_byrepeat[j][0]
            # get an estimator
            model = util.graphs_to_linmodel(task.pos+pos.graphs, task.neg+neg.graphs)

            X,y= util.graphs_to_Xy(task.postest,task.negtest)

            asd=Counter( model.predict(X)*y )
            means.append( float( asd[1] ) / sum(asd.values()) )

        RESULT.append( (k, np.array(means).mean()))
    RESULT.sort()
    return RESULT



