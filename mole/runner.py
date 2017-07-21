import dill
import clean_make_tasks as make
import clean_sample
import clean_eval
import os
'''
the plan:
1. read the files
2. determine number of calc items
3. take cmd arg to choose a calc item
4. calc and save
'''

#  read tasks
def load_tasks():
    return dill.load(open("tasks","rb"))


# read results
def filename(task):
    return "out_%d_%d_%d" % (task.samplerid, task.size, task.repeat)

def load_result(task):
    print "reading", filename(task)
    return dill.load( open( filename(task),"rb") )


def collect_res():
    a  = load_tasks()
    return map(load_result,a)


def make_graphs():
    res= collect_res()
    processed = clean_eval.eval(res, dill.load(open("esti",'rb')))
    clean_eval.draw(processed, lambda x:x.score_mean , lambda x:x.score_var,'molescore.png')
    clean_eval.draw(processed, lambda x:x.time_mean , lambda x:x.time_var,'moletime.png')


def find_missins():
    a = load_tasks()
    missing=[]
    for i, task in enumerate(a):
        if os.path.isfile( filename(task) ) == False:
            missing.append(i+1)
    print "%d items are missing" % len(missing)
    for e in missing:
        print "qsub -t %d;" % e

def get_task(taskid):
    """i hope this destroys the object after use..."""
    a = load_tasks()
    return a[taskid]


repeats=3
train_sizes=[20,50,100,200,400,600,800,1000]
assid = '651610'

if False: # DEBUG :)
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
        a = list(make.make_data(assid,train_sizes=train_sizes,repeats=repeats))
        dill.dump(a[:-1],open("tasks","wb"))
        dill.dump(a[-1],open("esti","wb"))
        print 'number of tasks:',len(a)-1
        exit()

    if  sys.argv[1] == 'count':
        tasks = load_tasks()
        print len(tasks)
        exit()
    
    if  sys.argv[1] == 'miss':
        find_missins()
        exit()

    if  sys.argv[1] == 'draw':
        make_graphs()
        exit()


    taskid = int(sys.argv[1])-1 # sge can not have id 0 :/
    task=get_task(taskid)
    task= a[taskid]
    res = clean_sample.runwrap(task)
    #sampled = namedtuple("sampled",'samplerid,size,repeat,time,graphs')
    dill.dump(res,open(filename(task),'wb'))




