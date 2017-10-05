import toolz
import util
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from collections import Counter
from scipy.optimize import curve_fit

import sklearn

import layercomparison as lc






def run(fname,idd):
    task = util.loadfile(fname)[idd]
    #draw.graphlearn(decomposers[:5],size=10)
    esti= util.graphs_to_linmodel( task.pos, task.neg  )
    X,y = util.graphs_to_Xy(task.postest, task.negtest)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)
    util.dumpfile((task.size,acc), "ASD/d_%s_%d" % (fname,idd))


def run2(fname,idd):
    task = util.loadfile(fname)[idd]
    #draw.graphlearn(decomposers[:5],size=10)
    pos,neg=util.sample_pn(task)

    esti= util.graphs_to_linmodel( task.pos+pos.graphs, task.neg+neg.graphs  )
    X,y = util.graphs_to_Xy(task.postest, task.negtest)
    ypred = esti.predict(X)
    acc = sklearn.metrics.accuracy_score(y,ypred)
    util.dumpfile((task.size,acc), "ASD/%s_%d" % (fname,idd))


def draw(tasknum):
    plt.figure(figsize=(15,5))


    # draw the run points
    res=[util.loadfile("ASD/d_stack_taskABC_%d" % i ) for i in range (tasknum)]

    def res_to_xy(res):
        res= toolz.groupby(lambda x:x[0],res)
        keys=res.keys()
        keys.sort()
        print res[keys[0]]
        def avg(r):
            r=[ e[1] for e in r ]
            return sum(r)/float(len(r))
        y = [ avg(res[k]) for k in keys]
        return keys,y


    res2=[util.loadfile("ASD/stack_taskABC_%d" % i ) for i in range (tasknum)]


    x,y= res_to_xy(res)
    plt.plot(x,y,color='blue')

    x,y= res_to_xy(res2)
    plt.plot(x,y,color='red')

    plt.show()





if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run( sys.argv[1], int(sys.argv[2]) -1)
    run2( sys.argv[1], int(sys.argv[2]) -1)








