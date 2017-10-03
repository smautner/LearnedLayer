import matplotlib
matplotlib.use('Agg') 

from moleLearnedLayer import layercomparison as lc 
import layercomp_maketask as lm 



if __name__=="__main__":
    for i in range(5): # 5
        fname = "%s_%d%d" % (lm.aid, max(lm.sizes),i) # eg: bursi_8000  should be lm.aid
        lc.evalandshow( fname,  3*lm.repeats*len(lm.sizes),lm.sizes,show=False )
