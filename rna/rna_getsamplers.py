from graphlearn01.minor.rna.rnadecomposer import RnaDecomposer
import graphlearn01.minor.rna.infernal as infernal
from graphlearn01.minor.rna import forgitransform as forgitransform
from graphlearn01.learnedlayer import cascade
from graphlearn01.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar as grammar
from graphlearn01.minor.rna import get_sequence


# moved the class outside the make_samplers_rna function because dill died
class rna_default_sampler(infernal.AbstractSampler):
    def _return_formatter(self,graphlist,mon):
        def graph_to_rna(graph):
            return ('',get_sequence(graph))
        def graph_to_rna(graph):
            return ('',graph.graph['sequence'])
        self.monitors.append(mon)
        yield map(graph_to_rna, graphlist)
            
def make_samplers_rna(n_jobs=1):

    # default sampla
    sampler=rna_default_sampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer=forgitransform.GraphTransformerForgi(fold_only=True),
                            #decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            n_jobs=n_jobs,
                            include_seed=False
                           )
    samplers=[sampler]
    # forgi transformer
    # check if calc_contracted_edge_nodes false is not breaking things
    sampler=infernal.AbstractSampler(
                            #radius_list=[0,1],
                            #thickness_list=[2],
                            backtrack=4,
                            grammar = grammar(
                                min_cip_count=1,
                                min_interface_count=2),
                            n_jobs=n_jobs,
                            select_cip_max_tries=50,
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
                            n_jobs=n_jobs,
                            #min_cip_count=1,
                            #min_interface_count=2,
                            graphtransformer= cascade.RNACascade(),
                            decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True,calc_contracted_edge_nodes=True),
                            #estimator=estimator
                            #feasibility_checker=feasibility
                            include_seed=False
                           )

    samplers.append(sampler)
    return samplers
    




