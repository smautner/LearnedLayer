import dill
import clean_make_tasks as make
import clean_sample
import clean_eval

'''
the plan:
1. read the files
2. determine number of calc items
3. take cmd arg to choose a calc item
4. calc and save
'''


#  read tasks
def load_tasks():
    size_rep_list, esti,make_samplers_chem = dill.load(open("tasks","rb"))
    return size_rep_list,esti,make_samplers_chem


def count_tasks(size_rep,samplers):
    res=[]
    for i in range(len(samplers)): # samplers
        for j in range(len(size_rep)): # sizes
            for k in range(len(size_rep[0])): # repeats are the same for all sizes
                res.append( (i,j,k) )
    return res


# read results
def filename(t):
    return "out_%d_%d_%d" % tuple(t)

def load_result(se,i,j):
    print filename((se,i,j))
    return dill.load( open( filename((se,i,j)),"rb") )


def collect_res(size_rep,samplers):
    res= [[[ load_result(se,i,j) for j in range(len(size_rep[0]))  ]
        for i in range(len(size_rep))]
            for se in range(len(samplers))]
    return res




def make_graphs(train_sizes):
    li,es,sa = load_tasks()
    res= collect_res(li,sa)
    clean_eval.eval(res,es, train_sizes)



def find_missins():
    l,e,sa=load_tasks()
    t= count_tasks(l,sa)
    import os
    missing=[]
    for i, tri in enumerate(t):
        if os.path.isfile( filename(tri) ) == False:
            missing.append(i)
    print "%d items are missing" % len(missing)
    for e in missing:
        print "qsub -t %d;" % e



repeats=3
train_sizes=[20,50,100,200,400,600,800,1000]
assid = '651610'

if True: # DEBUG :)
    assid = '1834'
    train_sizes=[20,100]
    repeats =2

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
        -- maketasks
        """    
        exit()


    if sys.argv[1] == 'maketasks':
        a = make.get_tasks(assid,train_sizes=train_sizes,repeats=repeats)
        a[2]= a[2][:2] # Only 2 here :)
        dill.dump(a,open("tasks","wb"))
        print 'number of tasks:',len(count_tasks(a[0],a[2]))
        exit()

    if  sys.argv[1] == 'count':
        l,e,sa= load_tasks()
        print len(count_tasks(l,sa))
        exit()
    
    if  sys.argv[1] == 'miss':
        find_missins()
        exit()

    if  sys.argv[1] == 'draw':
        make_graphs(train_sizes)
        exit()


    l,e,sa = load_tasks()
    taskid = int(sys.argv[1])-1 # sge can not have id 0 :/
    t= count_tasks(l,sa)[taskid]

    samplerid, sizeid, repeat = t
    sampler= sa[samplerid]
    data   = l[sizeid][repeat]

    res = clean_sample.runwrap(sa[samplerid], data)
    dill.dump(res,open(filename(t),'wb'))


######
### put arg options to find missind and make_graphs
#####


