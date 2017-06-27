




'''
the plan:
1. read the files
2. determine number if calc items
3. take cmd arg to choose a calc item
4. calc and save
'''

import dill
def get_data():
    data = dill.load(open("data","rb"))
    sampler = dill.load(open("samplers","rb"))
    return data,sampler

def task_count(data,samplers):
    res=[]
    for i in len(samplers):
        for j,d in enumerate(data):
            for k in len(d):
                res.append( (i,j,k) )
    return res





if __name__ == '__main__':
    import sys
    taskid = int(sys.argv[1])

    d,s=get_data()
    t= task_count(d,s)
    print len(t)

    print taskid

