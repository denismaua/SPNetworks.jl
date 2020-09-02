# Algorithms for Maximum-a-posteriori inference

module MAP

using SPNetworks
import SPNetworks: 
    Node, SumNode, ProductNode, LeafNode, CategoricalDistribution, IndicatorFunction, GaussianDistribution,
    isleaf, isprod, issum,
    logpdf!,
    vardims

include("MaxProduct.jl")
include("LocalSearch.jl")
include("BeliefPropagation.jl")

end