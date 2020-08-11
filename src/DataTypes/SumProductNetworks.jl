# Implements SPN data type and associated helper functions

export
    SumProductNetwork,
    scope


"""
    SumProductNetwork(nodes::Vector{Node})

Implements a Sum-Product Network. 
Assumes nodes are numbered topologically so that `nodes[1]` is the root (output) of the network and nodes[end] is a leaf.

# Arguments
- `nodes`: vector of nodes sorted in topological order (use sort!(spn) after creating the network if this is not the case).
"""
struct SumProductNetwork
    nodes::Vector{Node}
end

"""
    length(spn::SumProductNetwork)

Computes the number of nodes in the sum-product network `spn`.
"""
Base.length(spn::SumProductNetwork) = length(nodes(spn))

"""
Get node by index.
"""
Base.getindex(spn::SumProductNetwork, i...) = getindex(nodes(spn), i...)

"""
    Traverses network (assumes nodes are topological ordered).
"""
function Base.iterate(spn::SumProductNetwork, state = 1)
    if state > length(spn)
        return nothing
    end
    return (spn[state],state+1)
end
Base.eltype(spn::Type{SumProductNetwork}) = Node

"""
    sort!(spn::SumProductNetwork)

Sort nodes in topological order (with ties broken by breadth-first order) and modify node ids accordingly.
Returns the permutation applied.
"""
function sort!(spn::SumProductNetwork)    
    # First compute the number of parents for each node
   pa = zeros(length(spn))
   for (i,n) in enumerate(spn)
        if !isleaf(n)
            for j in n.children            
                pa[j] += 1
            end
        end
    end
    @assert count(isequal(0), pa) == 1 "SumProductNetworks has more than one parentless node"
    root = findfirst(isequal(0),pa) # root is the single parentless node
    # Kanh's algorithm: collect node ids in topological BFS order
    open = Vector{Int}()
    # visited = Set{Int}()
    closed = Vector{Int}() # bfs order
    push!(open, root) # enqueue root node 
    while !isempty(open)
        n = popfirst!(open) # dequeue node
        # push!(visited, n)
        push!(closed, n)
        if !isleaf(spn[n])
            # append!(open, ch for ch in spn[n].children if !in(ch, visited) && !in(ch, open))
            for j in spn[n].children
                pa[j] -= 1
                if pa[j] == 0
                    push!(open, j)
                end
            end
        end
    end
    @assert length(closed) == length(spn)
    inverse = similar(closed) # inverse mapping
    for i=1:length(closed)
        inverse[closed[i]] = i
    end
    # permute nodes according to closed
    permute!(spn.nodes,closed) # is this faster than spn.nodes .= spn.nodes[closed]? 
    # now fix ids of children
    for i=1:length(spn)
        if !isleaf(spn.nodes[i])
            for (j,ch) in enumerate(spn.nodes[i].children)
                spn.nodes[i].children[j] = inverse[ch]
            end
        end
    end
    closed
end
"""
    nodes(spn::SumProductNetwork)

Collects the list of nodes in `spn`.
"""
@inline nodes(spn::SumProductNetwork) = spn.nodes


"""
Select nodes by topology
"""
@inline leaves(spn::SumProductNetwork) = filter(n -> isa(n, LeafNode), nodes(spn)) 
@inline sumnodes(spn::SumProductNetwork) = filter(n -> isa(n, SumNode), nodes(spn)) 
@inline productnodes(spn::SumProductNetwork) = filter(n -> isa(n, ProductNode), nodes(spn)) 
@inline root(spn::SumProductNetwork) = @inbounds spn.nodes[1]
#TODO #variables(spn::SumProductNetwork) = collect(1:spn._numvars)
@inline children(spn::SumProductNetwork,n) = @inbounds spn.nodes[n].children
"""
Return vector of weights associate to outgoing edges of (sum) node n.
"""
@inline weights(spn::SumProductNetwork, n) = @inbounds spn.nodes[n].weights
# """
# Return the value of the weight associate to edge from `i` to `j`
# """
#@inline getweight(spn::SumProductNetwork, i, j) = @inbounds spn.nodes[n].weights[j]
# """
# Return the log-value of the weight associate to edge from `i` to `j`
# """
#@inline logweight(spn::SumProductNetwork, i, j) = log(getweight(spn,i,j))
"""
    size(spn)

Computes the number of parameters in the network `spn`.
"""
function Base.size(spn::SumProductNetwork)
    numparams = 0
    for i = 1:length(spn)
        if isa(spn[i],SumNode)
            numparams += length(children(spn,i))
        elseif isa(spn[i],CategoricalDistribution)
            numparams += length(spn[i].values)
        elseif isa(spn[i],GaussianDistribution)
            numparams += 2
        end
    end
    numparams
end

"""
    scope(spn)

Returns the scope of network `spn`, given by the scope of its root node.
"""
scope(spn::SumProductNetwork)::AbstractVector = unique(collect(map(n->scope(n), leaves(spn))))  
