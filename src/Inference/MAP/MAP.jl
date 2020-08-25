# Algorithms for Maximum-a-posteriori inference

module MAP

using SumProductNetworks
import SumProductNetworks: 
    Node, SumNode, ProductNode, LeafNode, CategoricalDistribution, IndicatorFunction, GaussianDistribution,
    isleaf, isprod, issum,
    logpdf!

include("MaxProduct.jl")
include("LocalSearch.jl")
include("BeliefPropagation.jl")

end