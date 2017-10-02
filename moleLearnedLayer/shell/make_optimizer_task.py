


from moleLearnedLayer import optimize, util
aid='651610'
size=100
repeats=4


# these options are read by the eval thing.. 
typ=1 
num_tries=100 

if __name__ == "__main__":
    data = util.init_optimisation(aid=aid,size=size,repeats=repeats)
    print data, typ
    print "t:", num_tries
