using SumProductNetworks
using Test
using BenchmarkTools

@info "# SumProductNetworks.jl Package"

# Creation, evaluation, sampling
# include("testspn.jl")
# include("testgspn.jl")
# include("testeval.jl")

# I/O
include("testio.jl")             

# MAP Inference
# include("testmaxproduct.jl")
# include("testspn2milp.jl")

# Parameter learning
# include("testparamlearn.jl")
# include("testparamlearngspn.jl")

# Structure learning
# include("testlearnspn.jl")
# include("testchowliu.jl")

# Dense network generation
# include("testchain.jl")
# include("testdense.jl")
