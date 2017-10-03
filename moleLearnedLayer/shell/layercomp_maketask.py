from moleLearnedLayer import layercomparison as lc
from moleLearnedLayer import util

aid="bursi"
sizes=[50,100,200,300,400,600,800]
repeats=5
opts = [util.loadfile("TOP5_%d" % i) for i in range(3)  ]


if __name__ == "__main__":
    taskfiles=[]
    for i,(a,b,c) in enumerate(zip( *opts) ): 
        taskfiles.append( lc.make_task_file (aid,sizes,repeats, params=[a[1],b[1],c[1]], taskfile_poststring=str(i)))

    print taskfiles
    print 'tasknum: %d' % ( len(sizes)*repeats*3)
    print "see layercomp_run.sh"

