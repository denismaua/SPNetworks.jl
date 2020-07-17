# Implements the Max-Product Algorithm for MAP Inference

export
    maxproduct!

"""
    maxproduct!(evidence,spn,query)

Approximates the maximum value of a configuration of query variables for given some evidence. 

# Parameters
- `evidence`: vector of values for each variable; NaN values denote sum-out variables; the algorithm fills-in the values of query variables. 
- `spn`: sum-product network.
- `query`: set of variables to be maximized.
"""
function maxproduct!(evidence::AbstractVector,spn::SumProductNetwork,query::AbstractSet)
    values = Vector{Float64}(undef,length(spn))
    tree = Vector{UInt}(undef,length(spn))
    # run maxproduct
    mp = maxproduct!(values,tree,spn,query,evidence)
    # backtrack induced tree to find corresponding assignment
    stack = Vector{UInt}([1])
    while !isempty(stack)
        n = pop!(stack) # remove last
        node = spn[n]
        if isa(node,SumNode)
            push!(stack,tree[n]) # add selected child
        elseif isa(node,ProductNode)
            append!(stack,node.children)
        elseif node.scope in query # leaf node with query variable
            evidence[node.scope] = argmax(node)
        end
    end
    mp
end
"""
    maxproduct!(values,spn,query,evidence)

Approximates the maximum value of a configuration of query variables for given some evidence. 

# Parameters
- `values`: vector of node values (to be computed by algorithm)
- `tree`: vector represented induced tree selected by the algorithm
- `spn`: sum-product network
- `query`: set of variables to be maximized
- `evidence`: vector of values for each variable; NaN values denote sum-out variables and the values of query variables are ignored
"""
function maxproduct!(values::AbstractVector,tree::AbstractVector,spn::SumProductNetwork,query::AbstractSet,evidence::AbstractVector)
    @assert length(values) == length(spn) 
    # traverse nodes in reverse topological order (bottom-up)
    for i in length(spn):-1:1
        node = spn[i]
        if isa(node,ProductNode)
            logval = 0.0
            for j in node.children
                @inbounds logval += values[j]
            end
            @inbounds values[i] = isfinite(logval) ? logval : -Inf
        elseif isa(node,SumNode)
            maxval = -Inf
            argval = 0
            for (k,j) in enumerate(node.children)
                @inbounds val = values[j] + log(node.weights[k])
                if val > maxval
                    maxval = val
                    argval = j
                end
            end
            @inbounds values[i] = isfinite(maxval) ? maxval : -Inf
            @inbounds tree[i] = argval
        else # is a leaf node
            if node.scope in query
                @inbounds values[i] = log(maximum(node))
            else
                @inbounds values[i] = logpdf(node,evidence[node.scope])
            end
        end
    end
    return values[1]
end
