
import dill
import rna_getdata as getdata
import rna_run as runner
import os
import rna_eval
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


def make_graphs(rfamid):
    res= collect_res()
    processed = rna_eval.eval(res, rfamid)
    rna_eval.draw(processed, lambda x:x.score_mean , lambda x:x.score_var,'rna_score.png')
    rna_eval.draw(processed, lambda x:x.time_mean , lambda x:x.time_var,'rna_time.png')


def find_missins():
    a = load_tasks()
    missing=[]
    for i, task in enumerate(a):
        if os.path.isfile( filename(task) ) == False:
            missing.append(i+1)
    print "%d items are missing" % len(missing)
    for e in missing:
        print "qsub -t %d;" % e



repeats=3
train_sizes=[20,50,100,200,400]
rfamid = 'RF00005'

if False: # DEBUG :)
    train_sizes=[25,50]
    repeats =2

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:    # NO ARGS
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
        a = list(getdata.get_data(rfamid,trainsizes=train_sizes,repeats=repeats))
        dill.dump(a,open("tasks","wb"))
        print 'number of tasks:',len(a)
        exit()

    if  sys.argv[1] == 'count':
        tasks = load_tasks()
        print len(tasks)
        exit()

    if  sys.argv[1] == 'miss':
        find_missins()
        exit()

    if  sys.argv[1] == 'draw':
        make_graphs(rfamid)
        exit()


    a = load_tasks()
    taskid = int(sys.argv[1])-1 # sge can not have id 0 :/

    task= a[taskid]
    res = runner.run(task)
    #sampled = namedtuple("sampled",'samplerid,size,repeat,time,sequences')

    dill.dump(res,open(filename(task),'wb'))



