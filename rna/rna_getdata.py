import os.path
import random
from eden_rna.io.fasta import load
import itertools
from eden.util import selection_iterator
import urllib
from collections import namedtuple
import copy

import rna_getsamplers as gs





################
#  all this is to load sequences
###############
def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)

def rfam_uril(family_id):
    return '%s.fa'%(family_id)

def rfamhandler(famid):
    filename=rfam_uril(famid)
    if not os.path.isfile(filename):
        urllib.urlretrieve(rfam_uri(famid),filename)
    return filename

def get_sequences(rfamid='RF00005',size=9999,withoutnames=False):
    sequences = itertools.islice( load(rfamhandler(rfamid)), size)
    if withoutnames:
        return [ b for (a,b) in sequences ]
    return sequences

def getseqs(allsequences, amount):
    random.shuffle(allsequences)
    return allsequences[:amount]


#######
#  make the data
########
task = namedtuple("task",'samplerid size repeat sampler sequences')
def get_data(rfamid,repeats, trainsizes=[10,20]):
    for trainsize in trainsizes:
        for repeat in range(repeats):
            allseqs = list(get_sequences(rfamid))
            for i, sampler in enumerate(gs.make_samplers_rna(n_jobs=1)):
                yield task(i,trainsize,repeat,sampler,copy.deepcopy( getseqs(allseqs,trainsize) ))






