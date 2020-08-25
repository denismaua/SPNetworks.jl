# Implements a 1-Neighborhod Stochastic Local Search for MAP Inference

"""
    lsmax!(evidence,spn,query)

Performs a local search for the maximum value of a configuration of `query` variables for given some `evidence`. 
Returns the value and the last configuration visited by the search (the local optimum or maximum iterations limit).

# Parameters
- `evidence`: vector of values for each variable; NaN values denote sum-out variables; the algorithm fills-in the values of query variables. 
- `spn`: sum-product network.
- `query`: set of variables to be maximized.
"""
function lsmax!(evidence::AbstractVector,spn::SumProductNetwork,query::AbstractSet)
    values = Vector{Float64}(undef,length(spn))
    ls = localsearch!(values,evidence,spn,query)
end
"""
    lsmax!(values,tree,spn,query,evidence)

    Performs a local search for the maximum value of a configuration of `query` variables for given some `evidence`. 
Returns the value achieved at the last iteration of the search.

# Parameters
- `values`: vector of node values (to be computed by algorithm)
- `evidence`: vector of values for each variable; NaN values denote sum-out variables and the values of query variables are ignored
- `spn`: sum-product network
- `query`: set of variables to be maximized
- `maxiterations`: maximum number of iterations.
"""
function lsmax!(values::AbstractVector,evidence::AbstractVector,spn::SumProductNetwork,query::AbstractSet, maxiterations=100)
    @assert length(values) == length(spn) 
    best = logpdf!(values,evidence)
    vdims = vardims(spn)
    it = 1
    while it < maxiterations
        cur = best
        for var in query
            v = x[var]
            for i = 1:vdims[var]
                x[var] = i
                prob = logpdf!(values,spn,x)
                if prob > best
                    best = prob
                    v = i
                    break
                end
            end
            x[var] = v
        end
        if cur == best # local optimum reached
            break
        end
        it += 1
    end
    best
end
