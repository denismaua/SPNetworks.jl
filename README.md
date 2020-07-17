# SumProductNetworks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://denismaua.github.io/SumProductNetworks.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://denismaua.github.io/SumProductNetworks.jl/dev)
[![Build Status](https://github.com/denismaua/SumProductNetworks.jl/workflows/CI/badge.svg)](https://github.com/denismaua/SumProductNetworks.jl/actions)

# SumProductNetwork.jl

Yet another implementation of Sum-Product Networks in Julia.

# Features

## Representation

- Discrete SPNs with Categorical distributions and Indicator Functions at leaves
- Gaussian SPNs (Gaussian distributions at leaves)
- TODO: SPNs with Bayesian Trees at leaves

## Inference

- Marginal inference
- MAP Inference:
  - MaxProduct
  - TODO: ArgMaxProduct
  - TODO: SPN2MILP
  - TODO: Hybrid Message Passing

## Learning

- TODO: EM parameter learning for Categorical and Gaussian SPNs (numerically unstable)
- TODO: LearnSPN
- TODO: Mixture of Chow-Liu Trees

## Input/Output

- Load network from file
- Write network to file
- Write graphviz graphical representation (dot file format)

# Usage

```julia
    # Creating a simple categorical SPN from string/file
    io = IOBuffer("""# 
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
    SPN, totaltime = SumProductNetworks.load(io)
    @show SPN # show some information
    for a in 1:2, b in 1:2
        prob = SPN([a,b]) # compute probability of configuration
        println("SPN($a,$b) = $prob")
    end


    # Selective SPN
    @show selSPN = SumProductNetwork(
        [
         SumNode([2,3],[0.4,0.6]),              # 1
         ProductNode([4,5,6]),                  # 2
         ProductNode([7,8,9]),                  # 3
         IndicatorFunction(1,2.0),  # 4
         CategoricalDistribution(2,[0.3,0.7]),  # 5
         CategoricalDistribution(3,[0.4,0.6]),  # 6
         CategoricalDistribution(2,[0.8,0.2]),  # 7
         CategoricalDistribution(3,[0.9,0.1]),  # 8
         IndicatorFunction(1,1.0) # 9
        ]
    )
    # Computing marginal
    value = selSPN([1,NaN,NaN])
    println("selSPN(A=1) = $value") # ≈ 0.6
    # in log-domain
    @show logpdf(selSPN,[2,NaN,NaN]) # ≈ log(0.4)

    # MAP Inference
    evidence = [0.0,0.0,0.0] # no evidence, maximize all variables -- solution is stored in this vector
    mp = maxproduct!(evidence, selSPN, Set([1,2,3])) # evidence/solution, spn, query variables (all) 
    println("MaxProduct(A=$(evidence[1]),B=$(evidence[2]),C=$(evidence[3])) -> $(exp(mp))")
    @show logpdf(selSPN,evidence) 
```

# License

See LICENSE file.
