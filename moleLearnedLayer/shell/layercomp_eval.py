import matplotlib
matplotlib.use('Agg') 

from moleLearnedLayer import layercomparison as lc 
import layercomp_maketask as lm 


# usage: seq 0 4 | parallel python layercomp_eval.py 

import sys
if __name__=="__main__":
        i=sys.argv[1]
        fname = "%s/%d%s" % (lm.aid, max(lm.sizes),i) # eg: bursi/8000  should be lm.aid
        lc.evalandshow( fname,  3*lm.repeats*len(lm.sizes),lm.sizes,show=False )
