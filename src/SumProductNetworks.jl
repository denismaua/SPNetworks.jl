"""
# SumProductNetworks.jl

A package for manipulating Sum-Product Networks.

## Introduction

A Sum-Product Network (SPN) is either

  - a univariate distribution (leaf node), or
  - a convex combination of SPNs (sum node), or
  - a product of SPNs.

SPNs are represented as rooted graphs; internal nodes represent convex
combinations and products; leaves represent distributions.

"""
module SumProductNetworks

# required packages
#using SparseArrays
using Clustering

import Random

# Implements basic data types and helper functions
include("DataTypes/Nodes.jl")
include("DataTypes/SumProductNetworks.jl")

# Implements helper functions not dependent on SPNs
include("Utils.jl")

# For evaluating and sampling from SPN
include("Inference/Evaluation.jl")

# MAP Inference
include("Inference/MAP/MAP.jl")

# I/O functions
include("IO.jl")

# For learning the parameters of SPNs
# include("Learning/ParameterLearning.jl")

# For learning the structure of SPNs
# include("Learning/StructureLearning.jl")
include("DataTypes/BayesianNets.jl")

# For generating Dense SPNs (for use e.g. in images)
# include("Learning/DenseSPNs.jl")

end # end of module
