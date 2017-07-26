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
    return [get_default_sampler(n_jobs=n_jobs),get_hand_sampler(n_jobs=n_jobs), get_learned_sampler(n_jobs=n_jobs)]




def get_default_sampler(n_jobs=1, kwargs={}):
    # default sampla
    kwargs=kwargs.copy()

    grammaropts= kwargs.get('grammar_options',{})
    kwargs.pop("grammar_options",None)
    sampler=rna_default_sampler(
                            grammar=grammar(**grammaropts),
                            graphtransformer=forgitransform.GraphTransformerForgi(fold_only=True),
                            n_jobs=n_jobs,
                            include_seed=False,
                            **kwargs
                           )
    return sampler




def get_hand_sampler(n_jobs=1):
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
    return sampler



def get_learned_sampler(n_jobs=1, kwargs={}):

    kwargs=kwargs.copy()
    grammarargs=kwargs.pop("grammar_options",{})
    learnargs=kwargs.pop("learn_params",{})

    sampler=infernal.AbstractSampler(
                            grammar=grammar(**grammarargs),
                            n_jobs=n_jobs,
                            graphtransformer= cascade.RNACascade(**learnargs),
                            decomposer=RnaDecomposer(output_sequence=True,pre_vectorizer_rm_f=True,calc_contracted_edge_nodes=True),
                            include_seed=False,
                            **kwargs
                           )

    return sampler
    




