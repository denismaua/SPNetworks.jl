# SPNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://denismaua.github.io/SPNetworks.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://denismaua.github.io/SPNetworks.jl/dev)
[![Build Status](https://github.com/denismaua/SumProductNetworks.jl/workflows/CI/badge.svg)](https://github.com/denismaua/SPNetworks.jl/actions)

A Julia package for manipulating Sum-Product Networks.

## Introduction

A Sum-Product Network (SPN) is either

- a tractable distribution (e.g. a univariate discrete distribution), or
- a convex combination of SPNs (sum node), or
- a product of SPNs.

SPNs are represented as rooted graphs, internally represented as linked arrays.
The internal nodes represent convex combinations and products, and the leaves represent distributions.

## Features

### Representation

- Discrete SPNs with Categorical distributions and Indicator Functions at leaves
- Gaussian SPNs (Gaussian distributions at leaves)
- _TODO_: SPNs with Bayesian Trees at leaves

### Inference

- Marginal inference
- MAP inference:
  - MaxProduct
  - Hybrid Message Passing
  - Stochastic Local Search
  - _TODO_: ArgMaxProduct
  - SPN2MILP (but see package SPN2MILP for integration with Gurobi)

### Learning

- _TODO_: EM parameter learning for Categorical and Gaussian SPNs (numerically unstable)
- _TODO_: Structure learning by LearnSPN
- _TODO_: Mixture of Chow-Liu Trees

### Input/Output

- Load network from file
- Write network to file
- Write graphviz graphical representation (dot file format)

## Installation

In a julia shell, type `[` to enter the Pkg mode then type
```shell
# pkg> add https://github.com/denismaua/AlgebraicDecisionDiagrams.jl
pkg> add https://github.com/denismaua/GraphicalModels.jl
pkg> add https://github.com/denismaua/SPNetworks.jl
```
You can type `backspace` to leave the Pkg mode.

## Basic Usage

```julia
# Load package
using SPNetworks
# Creating a simple categorical SPN from string/file
io = IOBuffer("""# A simple SPN (anything after # is ignored)
# Inner nodes
1 + 2 0.2 3 0.5 4 0.3
2 * 5 7
3 * 5 8
4 * 6 8
# Leaves
5 categorical 1 0.6 0.4
6 categorical 1 0.1 0.9
7 categorical 2 0.3 0.7
8 categorical 2 0.8 0.2
#""")
spn = SumProductNetwork(io)
summary(spn) # show some information about network
for a in 1:2, b in 1:2
    prob = spn(a,b) # compute probability of configuration
    println("spn($a,$b) = $prob")
end

import SPNetworks: SumNode, ProductNode, CategoricalDistribution, IndicatorFunction
# Selective SPN
@show selspn = SumProductNetwork(
    [
        SumNode([2,3],[0.4,0.6]),              # 1
        ProductNode([4,5,6]),                  # 2
        ProductNode([7,8,9]),                  # 3
        IndicatorFunction(1,2.0),              # 4
        CategoricalDistribution(2,[0.3,0.7]),  # 5
        CategoricalDistribution(3,[0.4,0.6]),  # 6
        CategoricalDistribution(2,[0.8,0.2]),  # 7
        CategoricalDistribution(3,[0.9,0.1]),  # 8
        IndicatorFunction(1,1.0)               # 9
    ]
)
# Computing marginal probabilities
# Summed-out variables are represented as NaN assigmments:
value = selspn(1,NaN,NaN)
println("selspn(A=1) = $value") # ≈ 0.6
# or in log-domain
@show logpdf(selspn,[2,NaN,NaN]) # ≈ log(0.4)

# MAP Inference
import SPNetworks.MAPInference: maxproduct!
evidence = [0.0,0.0,0.0] # no evidence, maximize all variables -- solution is stored in this vector
query = Set([1,2,3]) # variables to be maximized (all) -- non-evidence, non-query variables are marginalized
mp = maxproduct!(evidence, selspn, query) # run maxproduct and store solution in evidence
println("MaxProduct(A=$(evidence[1]),B=$(evidence[2]),C=$(evidence[3])) -> $(exp(mp))")
@show logpdf(selspn,evidence) # compute log-probability of solution
```

## License

Copyright (c) 2020 Denis D. Mauá.

See LICENSE file for more information.

## Why SPNetworks?

There already exists a package names `SumProductNetworks`; besides, SP is the acronym of the city I live in, so that makes an interesting coincidence.
