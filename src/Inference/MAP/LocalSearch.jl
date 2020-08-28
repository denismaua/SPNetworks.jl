# Implements a 1-Neighborhod Stochastic Local Search for MAP Inference
import SumProductNetworks: vardims

"""
    localsearch!(evidence,spn,query)

Performs a local search for the maximum value of a configuration of `query` variables for given some `evidence`. 
Returns the value and the last configuration visited by the search (the local optimum or maximum iterations limit).

# Parameters
- `evidence`: vector of values for each variable; NaN values denote sum-out variables; the algorithm fills-in the values of query variables. 
- `spn`: sum-product network.
- `query`: set of variables to be maximized.
- `maxiterations`: maximum number of iterations [default: 100].
"""
function localsearch!(evidence::AbstractVector,spn::SumProductNetwork,query::AbstractSet,maxiterations=100)
    values = Vector{Float64}(undef,length(spn))
    ls = localsearch!(values,evidence,spn,query,maxiterations)
end
"""
    localsearch!(values,x,spn,query,maxiterations)

    Performs a local search for the maximum value of a configuration of `query` variables for given some `evidence`. 
Returns the value achieved at the last iteration of the search.

# Parameters
- `values`: vector of node values (to be computed by algorithm).
- `x`: vector of values for each variable; NaN values denote sum-out variables and the values of query variables are ignored.
- `spn`: sum-product network.
- `query`: set of variables to be maximized.
- `maxiterations`: maximum number of iterations.
"""
function localsearch!(values::AbstractVector,x::AbstractVector,spn::SumProductNetwork,query::AbstractSet, maxiterations)
    @assert length(values) == length(spn) 
    best = logpdf!(values,spn,x)
    vdims = vardims(spn)
    it = 1
    before = time_ns()
    while it < maxiterations
        now = time_ns()
        if (now-before) > 60*1e9
            # report incumbent result every minute
            printstyled("[$it/$maxiterations]"; color = :lightcyan)
            printstyled(" $best "; color = :normal)
            printstyled("  [$((now-before)*1e-9)s]\n"; color = :light_black)
            before = now
        end
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
