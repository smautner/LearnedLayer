
import numpy as np
import matplotlib.pyplot as plt
import toolz
from collections import namedtuple
from graphlearn01.minor.rna.infernal import infernal_checker
processed_result=namedtuple("processed_result","samplerid, size, score_mean, score_var, time_mean, time_var")


def getcol(procress):
    coldict={1:'#F94D4D',
             2:'#555555',
             0:'#6A9AE2'}
    return coldict[procress.samplerid]


def sequences_to_scores(seq, rfamid):
    # need an np array of stuff
    seq = infernal_checker(seq, cmfile='../tools/%s.cm' % rfamid , cmsearchbinarypath='../tools/cmsearch')
    return np.array(seq)


def eval(res,rfam):
    processed=[]
    for byrepeats in toolz.groupby(lambda x:x.samplerid+x.size, res).values():
        time=np.array([ x.time for x in byrepeats ])
        graphs = [g[1] for x in byrepeats for g in x.sequences ]
        scores = sequences_to_scores(graphs, rfam)
        processed.append(  processed_result(byrepeats[0].samplerid,byrepeats[0].size,scores.mean(),scores.var(),time.mean(),time.var())  )
    return processed

def draw(processed, get_mean, get_var,filename): # see runner :)
    plt.figure(figsize=(15,5))
    for oneline in toolz.groupby(lambda x:x.samplerid, processed).values():
        oneline.sort(key=lambda x:x.size)
        sizes = [x.size for x in oneline]
        y_values = np.array([get_mean(x) for x in oneline])
        y_variances= np.array([get_var(x) for x in oneline])
        col = getcol(oneline[0])
        plt.fill_between(sizes, y_values+y_variances , y_values -y_variances, facecolor=col, alpha=0.15, linewidth=0)
        plt.plot(sizes,y_values,color=col)
        plt.savefig(filename)

