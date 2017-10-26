import util
import numpy as np
import toolz
import graphlearn01.utils.draw  as draw
import sklearn
from matplotlib import pyplot as plt


def make_task(aid, sizes=[50,100,150], test=100,repeats=3, params={},postname=''):
    pos,neg = util.getgraphs(aid)
    tasks=[]
    for size in sizes:
        getsize=size+test
        for i,(p,n) in enumerate(util.sample_pos_neg(pos,neg,getsize,getsize,repeats)):
            sampler = util.get_casc_abstr(kwargs=params)
            tasks.append(util.task2(1,size,i,sampler,n[:size],p[:size],n[size:],p[size:]))

    util.dumpfile(tasks,"stack_task"+postname)






def run(fname,idd):
    def getacc(esti, a,b):
        X,y = util.graphs_to_Xy(a,b)
        ypred = esti.predict(X)
        acc = sklearn.metrics.accuracy_score(y,ypred)
        return acc

    task = util.loadfile(fname)[idd]
    # make an estimator with the full stack
    full_stacked = [task.sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in task.sampler.graph_transformer.fit_transform(task.pos,task.neg,
                        remove_intermediary_layers=False)]

    esti= util.graphs_to_linmodel( full_stacked[:task.size], full_stacked[task.size:]  )

    # fully stacked test instances
    testgraphs= [task.sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in task.sampler.graph_transformer.transform(task.postest+task.negtest,remove_intermediary_layers=False)]

    acc= getacc(esti, testgraphs[:len(task.postest)], testgraphs[len(task.postest):] )
    util.dumpfile((task.size,acc), "stacked/%s_%d" % (fname,idd))


    task = util.loadfile(fname)[idd]
    # make an estimator with the full stack
    full_stacked = [task.sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in task.sampler.graph_transformer.fit_transform(task.pos,task.neg,
                        remove_intermediary_layers=True)]

    esti= util.graphs_to_linmodel( full_stacked[:task.size], full_stacked[task.size:]  )

    # fully stacked test instances
    testgraphs= [task.sampler.decomposer.make_new_decomposer(data).pre_vectorizer_graph()
                   for data in task.sampler.graph_transformer.transform(task.postest+task.negtest,remove_intermediary_layers=True)]

    acc= getacc(esti, testgraphs[:len(task.postest)], testgraphs[len(task.postest):] )
    util.dumpfile((task.size,acc), "stacked/2_%s_%d" % (fname,idd))





def run2(fname,idd):
    task = util.loadfile(fname)[idd]
    #draw.graphlearn(decomposers[:5],size=10)
    esti= util.graphs_to_linmodel( task.pos, task.neg  )
    X,y = util.graphs_to_Xy(task.postest, task.negtest)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)

    util.dumpfile((task.size,acc), "stacked/s_%s_%d" % (fname,idd))

def draw(numtasks,taskfilename, show=True):
    plt.figure(figsize=(15,5))

    def plod(prefix='stacked/',col='red',label='nolabel'):

        res= [ util.loadfile("%s%s_%d" % (prefix,taskfilename,e)) for e in range(numtasks) ]
        rez= toolz.groupby(lambda x:x[0], res)
        keys=rez.keys()
        keys.sort()
        y_values=[]
        y_variances=[]
        for key in keys:
            values=np.array([ val for (crap, val) in rez[key]])
            #print values
            y_values.append(values.mean())
            y_variances.append(values.var())
        y_values=np.array(y_values)
        y_variances=np.array(y_variances)
        #print y_values, y_variances
        plt.fill_between(keys, y_values+y_variances , y_values -y_variances, facecolor=col,
                             alpha=0.15)
        plt.plot(keys,y_values,color=col,label=label)

    plod(label='All Layers')
    plod("stacked/s_",'blue',label='Default')
    plod("stacked/2_",'yellow',label='No Intermediate Layers')

    #plt.grid(color='r', linestyle='-', linewidth=2)  # this is how to grid appearently
    plt.legend(loc='lower right')
    plt.title("ACC Learned Layers")
    plt.ylabel("accuracy")
    plt.xlabel("Number of train graphs")
    plt.savefig("%s.png" % taskfilename)
    if show:
        plt.show()

def draw_combined(numtasks,taskfilenames=[],show=True):
    plt.figure(figsize=(15,5))

    def plod(prefix='stacked/',col='red',label='nolabel'):

        res= [ util.loadfile("%s%s_%d" % (prefix,taskfilename,e)) for e in range(numtasks) for taskfilename in taskfilenames ]
        rez= toolz.groupby(lambda x:x[0], res)
        keys=rez.keys()
        keys.sort()
        y_values=[]
        ##y_variances=[]
        perc25=[]
        perc75=[]
        for key in keys:
            values=np.array([ val for (crap, val) in rez[key]])
            #print values
            y_values.append(np.median(values))
            perc25.append(np.percentile(values,25))
            perc75.append(np.percentile(values,75))

            ##y_variances.append(values.var())
        y_values=np.array(y_values)
        ##y_variances=np.array(y_variances)
        #print y_values, y_variances
        plt.fill_between(keys, perc25 , perc75, facecolor=col,
                             alpha=0.15)
        plt.plot(keys,y_values,color=col,label=label)

    plod(label='All Layers')
    plod("stacked/s_",'blue',label='Default')
    plod("stacked/2_",'yellow',label='No Intermediate Layers')

    plt.grid(color='black', linestyle='-', linewidth=.5)  # this is how to grid appearently
    plt.legend(loc='lower right')
    plt.title("ACC Learned Layers")
    plt.ylabel("accuracy")
    plt.xlabel("Number of train graphs")
    plt.savefig("RAINBOW.png" )
    if show:
        plt.show()


if __name__ == '__main__':
    import sys
    run(sys.argv[1],int(sys.argv[2])-1)
    run2(sys.argv[1],int(sys.argv[2])-1)
