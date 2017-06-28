'''
the learned sampler has problems. 
we are here to fix them 
'''

import layerutils as lu

X,y,graphs,stats = lu.get_data("1834")

sampler = lu.get_casc_abstr()
sampler.graph_transformer.debug=True

res= sampler.fit_transform(graphs[:20])
res=list(res) # making sure to evaluate


