from graphlearn.minor.rna.rnadecomposer import RnaDecomposer
import graphlearn.minor.rna.infernal as infernal
from graphlearn.minor.rna import forgitransform as forgitransform
from graphlearn.learnedlayer import cascade
def make_samplers_rna():

    # default sampla
    sampler=infernal.AbstractSampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer=forgitransform.GraphTransformerForgi(fold_only=True),
                            #decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            include_seed=False
                           )
    samplers=[sampler]
    # forgi transformer
    # check if calc_contracted_edge_nodes false is not breaking things
    sampler=infernal.AbstractSampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer=forgitransform.GraphTransformerForgi(),
                            decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True,calc_contracted_edge_nodes=False),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            include_seed=False
                           )
    samplers.append(sampler)
    sampler=infernal.AbstractSampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer= cascade.RNACascade(),
                            decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True,calc_contracted_edge_nodes=False),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            include_seed=False
                           )

    samplers.append(sampler)
    return samplers
    




