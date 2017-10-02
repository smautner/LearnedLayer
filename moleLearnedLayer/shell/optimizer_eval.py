


from moleLearnedLayer import optimize , util
import make_optimizer_task as mot



taskfilename= "task_bursi_100_100_4"
for typ in [0,1,2]:
    top=optimize.report( taskfilename, typ, mot.num_tries, top=5  )
    util.dumpfile(top,"TOP5_%d" % typ)


