# Implements Bayesian networks

module BayesianNetworks

using SumProductNetworks
import SumProductNetworks: Node, SumNode, ProductNode, LeafNode, IndicatorFunction

"Tree-shaped Bayesian network. A tree is conneted digraph where each node has at most one parent."
abstract type BayesianTree end
struct BTRoot <: BayesianTree # root node
    variable::Int
    factor::Vector{Float64} # array of unconditional probabilities
    children::Vector{BayesianTree}
end
struct BTNode <: BayesianTree # inner node
    variable::Int
    factor::Matrix{Float64} # square matrix where columns are conditional probability distributions given each value of the parent
    children::Vector{BayesianTree}
end
struct BTLeaf <: BayesianTree # leaf node
    variable::Int
    factor::Matrix{Float64}
end

"Compile Bayesian Tree into SPN."
function compile(root::BTRoot) 
    # init cache of conditional SPNs
    cache = Dict{Tuple{Int,Int,Int},Int}() # (node.var,parent.var,value) -> node id
    # init cache of indicators
    icache = Dict{Tuple{Int,Int},Int}()    # (node.var,value) -> node id
    # root node
    sumnode = SumNode(Vector{UInt}(undef,size(root.factor,1)),root.factor)
    # list of nodes
    nodelist = Node[ sumnode ]
    # apply Shannon-like decomposition recursively
    for value=1:size(root.factor,1)
        # Decompose on root.variable = value
        prodnode = ProductNode([])
        push!(nodelist, prodnode)
        sumnode.children[value] = length(nodelist) # set child id 
        push!(nodelist, IndicatorFunction(root.variable,value))
        push!(prodnode.children,length(nodelist))
        for child in root.children
            # recur on child
            nodeid = compile!(cache,icache,nodelist,child,root,value)
            push!(prodnode.children,nodeid)
        end
    end
    # return compiled SPN
    spn = SumProductNetwork(nodelist)
    # node ids are not topologically orders - sort in-place nodes in bfs-order
    SumProductNetworks.sort!(spn)
    # return sorted network
    spn
end
"
    compile!(cache::Dict{Tuple{Int,Int,Int},Int},icache::Dict{Tuple{Int,Int}},nodelist::Vector{Node},node::BayesianTree,parent,pavalue)

Compile subtree rooted at node and adds respective nodes to nodelist. Returns id of root node of subSPN.
Caches the node id given the parent id and value to avoid duplication.
"
function compile!(cache::Dict{Tuple{Int,Int,Int},Int},icache::Dict{Tuple{Int,Int}},nodelist::Vector{Node},node::BTLeaf,parent,pavalue)
    if haskey(cache,(node.variable,parent.variable,pavalue))
        return cache[(node.variable,parent.variable,pavalue)]
    end
    sumnode = SumNode(Vector{UInt}(undef,size(node.factor,1)),node.factor[:,pavalue])
    push!(nodelist, sumnode)
    cache[(node.variable,parent.variable,pavalue)] = length(nodelist)
    for value = 1:size(node.factor,1)
        push!(nodelist,IndicatorFunction(node.variable,value))
        sumnode.children[value] = length(nodelist) # set child id 
    end
    # return root node id
    cache[(node.variable,parent.variable,pavalue)]    
end
function compile!(cache::Dict{Tuple{Int,Int,Int},Int},icache::Dict{Tuple{Int,Int}},nodelist::Vector{Node},node::BTNode,parent,pavalue)
    if haskey(cache,(node.variable,parent.variable,pavalue))
        return cache[(node.variable,parent.variable,pavalue)]
    end
    sumnode = SumNode(Vector{UInt}(undef,size(node.factor,1)),node.factor[:,pavalue])
    push!(nodelist, sumnode)
    cache[(node.variable,parent.variable,pavalue)] = length(nodelist)
    for value=1:size(node.factor,1)
        prod = ProductNode([])
        push!(nodelist, prod)
        sumnode.children[value] = length(nodelist) # set child id 
        if haskey(icache,(node.variable,value)) # recover indicator if already created
            push!(prod.children, icache[(node.variable,value)])
        else # new indicator 
            push!(nodelist, IndicatorFunction(node.variable,value))
            icache[(node.variable,value)] = length(nodelist) 
            push!(prod.children, length(nodelist))
        end
        for child in node.children
            # recur
            nodeid = compile!(cache,icache,nodelist,child,node,value)                
            push!(prod.children, nodeid)
        end
    end
    # return root node id
    cache[(node.variable,parent.variable,pavalue)]
end

end