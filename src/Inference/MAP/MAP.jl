# Algorithms for Maximum-a-posteriori inference

module MAP

using SumProductNetworks
import SumProductNetworks: 
    Node, SumNode, ProductNode, LeafNode, CategoricalDistribution, IndicatorFunction, GaussianDistribution,
    isleaf, isprod, issum

include("MaxProduct.jl")
include("BeliefPropagation.jl")
# include("SPN2MILP.jl")

end