

'''
write a task file
'''
import util

def make_task_file(aid='1834',sizes=[50,75,100],repeats=2):
    '''
    we drop lots of these in 1 file:
    task = namedtuple("task",'samplerid size repeat sampler neg pos')
    '''
    pos,neg = util.getgraphs(aid)
    tasks=[]
    for size in sizes:
        repeatsXposnegsamples = util.sample_pos_neg(pos,neg,size,size,repeats)
        for i, sampler in enumerate(util.get_all_samplers()):
            for j, (pos_sample, neg_sample) in enumerate(repeatsXposnegsamples):
                tasks.append(util.task(i,size,j,sampler,neg_sample,pos_sample))

    util.dumpfile(tasks,"%s_%d" % (aid,max(sizes)))


def showtask(filename, taskid):
    tasks= util.loadfile(filename)
    print tasks[taskid]

def run(filename, taskid):
    tasks= util.loadfile(filename)
    try:
        result = util.sample(tasks[taskid])
    except Exception as exc:
        print tasks[taskid]
        import traceback
        print traceback.format_exc(20)
        return None
    util.dumpfile(result,"res_%s_%d" % (filename,taskid))


if __name__ == '__main__':
    # yep optimize.py GRAPHFILE TYPE ID
    import sys
    run(sys.argv[1] , int(sys.argv[2]))







