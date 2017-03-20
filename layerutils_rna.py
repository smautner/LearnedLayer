from graphlearn.minor.rna.rnadecomposer import RnaDecomposer
import graphlearn.minor.rna.infernal as infernal
from graphlearn.minor.rna import forgitransform as forgitransform

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

    # forgi transformer
    sampler=infernal.AbstractSampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer=forgitransform.GraphTransformerForgi(),
                            decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            include_seed=False
                           )

    # learned samplari!
