'''
the learned sampler has problems. 
we are here to fix them 
'''

import dill
import moleLearnedLayer.clean_sample

X,y,graphs,stats = moleLearnedLayer.clean_make_tasks.get_data("1834")
d1,d2= moleLearnedLayer.clean_make_tasks.make_data("1834", repeats=1, train_sizes=[20, 100])
d1=d1[0]
d2=d2[0]

sampler = moleLearnedLayer.clean_make_tasks.get_no_abstr()

#sampler.graph_transformer.debug=True


if False: # save
    res= moleLearnedLayer.clean_sample.runwrap2(sampler, d1['graphs_train'], d1['train_graphs_neg'])
    res2= moleLearnedLayer.clean_sample.runwrap2(sampler, d2['graphs_train'], d2['train_graphs_neg'])

    with open("debugdata","wb") as file:
        dill.dump([res,res2], file)
else:

    with open("debugdata","rb") as file:
        res,res2= dill.load(file)


r1= d1['oracle'].decision_function(moleLearnedLayer.clean_make_tasks.vectorize(res[0]))
r2= d2['oracle'].decision_function(moleLearnedLayer.clean_make_tasks.vectorize(res2[0]))

print r1.mean()
print r2.mean()


"""
plan to redo things simpler:
1. the super mega list will look like this
V sampler V number V repeats V (graphs,time)

2. evaluate should take the run for 1 sampler in input 
and give me data to draw ONE line. 
// run a second time with a time flag or something


aaaargh also need to redo the plot thing..
 i imagine an object that first clears the pane,
 then add_line()
 then print/show() 
"""

