using Test
using BenchmarkTools

using SumProductNetworks
import SumProductNetworks: SumNode, ProductNode, CategoricalDistribution, IndicatorFunction

@testset "SumProductNetworks.jl" begin

@testset "Creation, evaluation, sampling" begin
    include("testspn.jl")
    # include("testgspn.jl")
    # include("testeval.jl")
end

# I/O
# include("testio.jl")             

# MAP Inference
@testset "MAP Infrence" begin
    # include("testmaxproduct.jl")
    # include("testbeliefprop.jl")
    # include("testspn2milp.jl")
end

# @testset "Parameter learning" begin
#   include("testparamlearn.jl")
#   include("testparamlearngspn.jl")
# end

# @testset "Structure learning" begin
#   include("testlearnspn.jl")
  # include("testchowliu.jl")
# end

# @testset "Dense network generation" begin
#   include("testchain.jl")
#   include("testdense.jl")
# end

end