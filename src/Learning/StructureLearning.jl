# Structure Learning

"""
    chainruleSPN(cardinalities::AbstractVector,depth::Int64=1,root::Variable=1)::SumProductNetwork

Generates a sum-product network encoding a naive joint distribution factorization according to the application of the Chain Rule for finite-valued variables with given `cardinalities`. The network is rooted at `root` node id, and performs a recursive decomposition starting at variable `depth`.

# Arguments

- `depth::Integer`: id of variable to decompose at.
- `root::Variable`: id of the root node of the SPN.

"""
function chainruleSPN(cardinalities::AbstractVector,depth::Integer=1,root::Integer=1)::SumProductNetwork
    cardinality = cardinalities[depth]
    nodelist = Vector{Node}()
    if depth == length(cardinalities)
        # final layer
        push!(nodelist,SumNode([i+root for i=1:cardinality],ones(cardinality)/cardinality))
        for value = 1:cardinality
            push!(nodelist,IndicatorFunction(depth,value))
        end
        return SumProductNetwork(nodelist)
        # return SumProductNetwork(Vector([CategoricalDistribution(depth,ones(cardinality)/cardinality)]))
    end
    push!(nodelist,SumNode([i+root for i=1:cardinality],ones(cardinality)/cardinality))
    index = root + cardinality + 1
    children = Vector{Node}()
    for value = 1:cardinality
        # Add product node with left child Indicator X=value and Left child the recursion
        push!(nodelist,ProductNode([index,index+1]))
        push!(children,IndicatorFunction(depth,value))
        index += 1
        # Now recur to build subSPNs
        S = chainruleSPN(cardinalities,depth+1,index)
        append!( children, nodes(S) )
        index += length(S)
    end
    append!(nodelist,children)
    SumProductNetwork(nodelist) 
end

"""
    fit(node, Data; params)

Learns (smoothed) Maximum Likelihood Estimators of univariate distribution from vector Data of i.i.d. samples. 
"""
function fit(node::LeafNode, Data::AbstractVector; params...)
    if isa(node,CategoricalDistribution)
        for datum in Data
           node.values[Int(datum)] += 1 
        end
        node.values ./= length(Data)
    else
        @error "Learning of this type of node not implemented"
    end

end


# TODO: Implement correlation matrix and splitting 
# This part was largely inspired by https://github.com/trappmartin/SumProductNetworks.jl/blob/master/src/structureUtilities.jl

"""
    splitrows(Data, scope, instances, minsamplesize)

Learns a sum node by clustering instances of columns in scope into two groups by k-means clustering. Returns a list of pairs (cluster,weight) where cluster is a vector of indexes view of a cluster and weights is the proportion of instances in that cluster.
"""
function splitrows(Data::AbstractMatrix, scope, instances, minsamplesize)
    @assert length(instances) > minsamplesize "Sample size too small for clustering"
    data = view(Data, instances, scope)
    results = kmeans(transpose(data), 2)
    cluster1 = instances[results.assignments .== 1] # view(Data, results.assignments .== 1, :)
    cluster2 = instances[results.assignments .!= 1] # view(Data, results.assignments .!= 1, :)
    weight = length(cluster1)/length(instances)

    if length(cluster1) < minsamplesize || length(cluster2) < minsamplesize
        return ((instances,1.0),)
    end
    return ((cluster1, weight), (cluster2, 1.0 - weight))
end

"""
    splitcolumns(Data, scope, instances, minsamplesize)

Learns a product node by clustering columns using the connected components of the association matrix after filtering out low score edges. Returns list of scopes of child nodes.
"""
function splitcolumns(Data::AbstractMatrix, scope, instances, minsamplesize)
    @assert length(scope) > 1 "Scope has size ($length(scope)); Must have at least two variables to split columns"
    if length(instances) < minsamplesize
        # stopping criterion, too small sample size: factorize
        return ((variable,) for variable in scope)
    else
        # generate random binary partition
        leftscope = scope[Random.randperm(length(scope))[1:floor(Integer,length(scope)/2)]]
        rightscope = setdiff(scope,leftscope)
        return (leftscope, rightscope)
    end
end

"""
    learnspn(Data,cardinalities,minsamplesize)

Implement Gen and Domingos 2013 top-down learning algorithm.
"""
function learnspn(Data::AbstractMatrix,cardinalities,minsamplesize=100)::SumProductNetwork

    numrows,numcolumns = size(Data)
    queue = Vector{Tuple}()
    nodes = Vector{Node}([])

    children = []
    to = Int64[]
    from = Int64[]
    weights = Float64[]

    # Start with a sum node as root
    # push!(queue, (1,1.0,collect(1:numcolumns),view(Data,:,:)))
    parent,scope,instances = 1,collect(1:numcolumns),collect(1:numrows)
    push!(nodes,SumNode(scope)) # add node to list
    node = 1 # node id 
    push!(children,[])
    for (cluster,weight) in splitrows(Data, scope, instances, minsamplesize)
        push!(queue,(node,weight,scope,cluster)) # enqueue
    end
    
    while length(queue) > 0

        parent,weight,scope,instances = popfirst!(queue) # dequeue
        #parent,weight,scope,dataset = pop!(queue) # remove from top

        # Stopping criterion, create univariate leaf
        if  length(scope) == 1
            leaf = CategoricalDistribution(scope[1],zeros(cardinalities[scope[1]]))
            fit(leaf,view(Data,instances,scope[1])) # learn node distribution from data
            push!(nodes,leaf) # add node to node list (node id is list size after insertion)
            push!(children[parent],length(nodes)) # add arc parent -> leaf
            push!(children,[]) # add outbound arcs (none)
            if isa(nodes[parent],SumNode)
                push!(from,parent)
                push!(to,node)
                push!(weights,weight) # add weight to arc parent -> node
            end
        elseif isa(nodes[parent],ProductNode)
            # Create sum node
            push!(nodes,SumNode(scope)) # add node to list
            node = length(nodes) # node id is node list size 
            push!(children,[])
            push!(children[parent],node)
            for (cluster,weight) in splitrows(Data, scope, instances, minsamplesize)
                push!(queue,(node,weight,scope,cluster)) # enqueue
            end
        else # isa(nodes[parent],SumNode)
            # Create product node
            push!(nodes,ProductNode(scope)) # add node to list
            node = length(nodes) # node id
            push!(children[parent],node)
            push!(children,[])
            push!(from,parent)
            push!(to,node)
            push!(weights,weight) # add weight to arc parent -> node
            for subscope in splitcolumns(Data, scope, instances, minsamplesize)
                push!(queue,(node,NaN,subscope,instances)) #enqueue
            end
        end

    end

    weightmatrix = sparse(to,from,weights) 
    
    spn = SumProductNetwork(nodes, children, weightmatrix)

    return spn
end
