from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pylab as plt

import util


def draw_distance_dummy():

    graphs , _  = util.getgraphs("bursi")


    sizes=[50,100,200]


    gen_train=[]
    test_train=[]
    for size in sizes:
        gen, test, train = graphs[:size], graphs[size:size*2], graphs[size*2:size*3]
        gen_train.append(get_distance(gen,train))
        test_train.append(get_distance(test,train))

    plot_stats(sizes,[gen_train,test_train], ylabels=['gen vs train','test vs train'])

def plot_stats(x=None, ylist=None, xlabel='', ylabel='', title='median distance to nearest neighbor', ylabels=['','']):
    plt.figure(figsize=(12, 3))

    colors={0:'red', 1:'blue'}
    for i,(y,label) in enumerate(zip(ylist, ylabels)):
        y = np.array(y)
        y0 = y[:, 0]
        y1 = y[:, 1]
        y2 = y[:, 2]
        plt.fill_between(x, y1, y2, alpha=0.2,color = colors[i])
        plt.plot(x, y0, 'o-', lw=2, label=label, color=colors[i])

    plt.xticks(x)
    plt.grid(linestyle=":")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def get_distance(seta, setb):
    seta=util.vectorize(seta)
    setb=util.vectorize(setb)
    nbrs = NearestNeighbors(n_neighbors=1).fit(seta)
    distances, indices = nbrs.kneighbors(setb)
    d=distances[:,0]
    return np.mean(d), np.percentile(d,25), np.percentile(d,75)



if __name__=="__main__":
    draw_distance_dummy()
