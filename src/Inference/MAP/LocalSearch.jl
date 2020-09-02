# Implements a 1-Neighborhod Stochastic Local Search for MAP Inference

"""
    localsearch!(evidence,spn,query)

Performs a local search for the maximum value of a configuration of `query` variables for given some `evidence`. 
Returns the value of the last configuration visited by the search (a local optimum or maximum iterations limit).

# Parameters
- `evidence`: vector of values for each variable with initial solution; NaN values denote sum-out variables; the algorithm overwrites the vector with the best solution found. 
- `spn`: sum-product network.
- `query`: set of variables to be maximized.
- `maxiterations`: maximum number of iterations [default: 100].
"""
function localsearch!(evidence::AbstractVector,spn::SumProductNetwork,query::AbstractSet,maxiterations=100)
    values = Vector{Float64}(undef,length(spn))
    best = logpdf!(values,spn,evidence)
    vdims = vardims(spn)
    it = 1
    before = time_ns()
    while it < maxiterations
        now = time_ns()
        if (now-before) > 60e9
            # report incumbent result every minute
            printstyled("[$it/$maxiterations]"; color = :lightcyan)
            printstyled(" $best "; color = :normal)
            printstyled("  [$((now-before)*1e-9)s]\n"; color = :light_black)
            before = now
        end
        cur = best
        for var in query
            v = evidence[var]
            for i = 1:vdims[var]
                evidence[var] = i
                prob = logpdf!(values,spn,evidence)
                if prob > best
                    best = prob
                    v = i
                    break
                end
            end
            evidence[var] = v
        end
        if cur == best # local optimum reached
            break
        end
        it += 1
    end
    best
end
