import dill
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=2)


'''
the plan:
0. write the tasks
1. read the files
2. determine number of calc items
3. take cmd arg to choose a calc item
4. calc and save
'''

'''write tasks'''
def write_task():
    from rna_getdata import get_data2
    from rna_getsamplers import make_samplers_rna

    data= get_data2("RF00005", repeats=2, trainsizes=[20,40,60])
    samplers = make_samplers_rna(n_jobs=2)
    dill.dump(data,open('data','wb'))
    dill.dump(samplers,open('samplers','wb'))



'''read tasks'''

import dill
def get_data():
    data = dill.load(open("data","rb"))
    sampler = dill.load(open("samplers","rb"))
    return data,sampler



def task_count(data,samplers):
    res=[]
    for i in range(len(samplers)):
        for j,d in enumerate(data):
            for k in range(len(d)):
                res.append( (i,j,k) )
    return res




'''
collecting results... 
'''


def load(se,i,j):
    print "out_%d_%d_%d" % (se,i,j)
    return dill.load( open("out_%d_%d_%d" % (se,i,j),"rb") )

def collect_res(d,s):
    res= [[[ load(se,i,j) for j in range(len(e)) ]
        for i,e in enumerate(d)]
            for se in range(len(s))]
    return res

def make_graphs():
    import layerutils as lu
    d,s = get_data()
    graphs_chem = collect_res(d,s)
    means,stds,means_time, stds_time = lu.evaluate(graphs_chem,d)
    # be careful with the train sizes.. check if they are right.. the obtaining is hacky
    lu.make_inbetween_plot(labels=lu.train_sizes, means=means , stds=stds,fname='chem.png')
    lu.make_inbetween_plot(labels=lu.train_sizes, means=means_time, stds=stds_time,fname='chem_time.png',dynamic_ylim=True)

def filename(t):
    return "out_%d_%d_%d" % tuple(t)

def find_missins():
    d,s=get_data()
    t= task_count(d,s)
    import os
    for i, tri in enumerate(t):
        if os.path.isfile( filename(tri) ) == False:
            print "%d is missing" % (i+1)



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:    # NO ARGS => count the tasks
        print "at least write help :)"
        exit()
    if  sys.argv[1] == 'help':
        print """
        -- count .. give number of tasks
        -- miss
        -- draw
        -- make .. make tasks
        """    
        exit()

    
    if  sys.argv[1] == 'miss':
        find_missins()
        exit()

    if  sys.argv[1] == 'draw':
        make_graphs()
        exit()

    if  sys.argv[1] == 'make':
        write_task()
        exit()

    d,s=get_data()
    #print d
    t= task_count(d,s)


    if  sys.argv[1] == 'count':
        print "there are %d tasks" % len(t)
        print t
        exit()


    taskid = int(sys.argv[1])-1 # sge can not have id 0 :/
    samplerid, d1,d2 = t[taskid]
    from rna_run import runner
    res = runner(s[samplerid],d[d1][d2])
    dill.dump(res,open("out_%d_%d_%d" % (samplerid,d1,d2),'wb'))


