# Routines for SPN evaluation and sampling

export
    logpdf, 
    rand

"""
    (spn::SumProductNetwork)(x::AbstractVector{<:Real})
    (spn::SumProductNetwork)(x...)

Evaluates the sum-product network at a given instantiation `x` of the variables.
Summed-out variables are represented as `NaN`s.

# Parameters

- `x`: vector of values of variables (integers or reals). 

# Examples

To compute the probability of ``P(b=1)`` using spn `S`, use
```julia
julia> S = SumProductNetwork(IOBuffer("1 + 2 0.2 3 0.5 4 0.3\n2 * 5 7\n3 * 5 8\n4 * 6 8\n5 categorical 1 0.6 0.4\n6 categorical 1 0.1 0.9\n7 categorical 2 0.3 0.7\n8 categorical 2 0.8 0.2"));

julia> S([NaN, 2])
0.3

julia> S(NaN, 2)
0.3
```
"""
function (spn::SumProductNetwork)(x::AbstractVector{<:Real})
    return exp(logpdf(spn,x))
end
function (spn::SumProductNetwork)(x...)
    return exp(logpdf(spn,[x...]))
end


"""
    logpdf(spn, X)

Returns the sums of the log-probabilities of instances `x` in `X`. Uses multithreading to speed up computations if `JULIA_NUM_THREADS > 1`.

# Parameters

- `spn`: Sum-Product Network
- `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as `NaN`s.
"""
function logpdf(spn::SumProductNetwork, X::AbstractMatrix{<:Real})::Float64
    # single-threaded version
    if Threads.nthreads() == 1
        values = Array{Float64}(undef,length(spn))
        return sum(logpdf!(values,spn,view(X,i,:)) for i=1:size(X,1))
    end
    # multi-threaded version
    values = Array{Float64}(undef,length(spn),Threads.nthreads())
    s = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i=1:size(X,1)
        Threads.atomic_add!(s, logpdf!(view(values,:,Threads.threadid()),spn,view(X,i,:)))
    end
    s[]
end

"""
    logpdf!(values, spn, X)

Computes log-probabilities of instances `x` in `X` and stores the results in a given vector. Uses multithreading to speed up computations.

# Parameters

- `results`: vector to store results (log-probabilities)
- `spn`: Sum-Product Network
- `X`: matrix of values of variables (integers or reals). Summed-out variables are represented as `NaN`s.
"""
function logpdf!(results::AbstractVector{<:Real}, spn::SumProductNetwork, X::AbstractMatrix{<:Real})
    @assert length(results) == size(X,1)
    # multi-threaded version
    values = Array{Float64}(undef,length(spn),Threads.nthreads())
    Threads.@threads for i=1:size(X,1)
        @inbounds results[i] = logpdf!(view(values,:,Threads.threadid()), spn, view(X,i,:))
    end
    Nothing
end

"""
    logpdf(spn, x)

Evaluates the sum-product network `spn` in log domain at configuration `x`.

# Parameters

- `x`: vector of values of variables (integers or reals). Summed-out variables are represented as `NaN`s

# Examples

To compute the probability of ``P(b=1)`` using spn `S`, use
```julia
julia> S = SumProductNetwork(IOBuffer("1 + 2 0.2 3 0.5 4 0.3\n2 * 5 7\n3 * 5 8\n4 * 6 8\n5 categorical 1 0.6 0.4\n6 categorical 1 0.1 0.9\n7 categorical 2 0.3 0.7\n8 categorical 2 0.8 0.2"));
julia> logpdf(S,[NaN, 2])
-1.2039728043259361

```
"""
function logpdf(spn::SumProductNetwork, x::AbstractVector{<:Real})::Float64
    values = Array{Float64}(undef,length(spn))
    return logpdf!(values,spn,x)
end


"""
    logpdf!(values,spn,x)

Evaluates the sum-product network `spn` in log domain at configuration `x` and stores values of each node in the vector `values`.
"""
function logpdf!(values::AbstractVector{Float64}, spn::SumProductNetwork, x::AbstractVector{<:Real})::Float64
    # @assert length(values) == length(spn)
    # traverse nodes in reverse topological order (bottom-up)
    @inbounds for i in length(spn):-1:1
        node = spn[i]
        if isprod(node)
            lval = 0.0
            for j in node.children
                lval += values[j]
            end
            values[i] = isfinite(lval) ? lval : -Inf
        elseif issum(node)
            # log-sum-exp trick to improve numerical stability
            m = -Inf
            # get maximum incoming value
            for j in node.children
                m = values[j] > m ? values[j] : m
            end
            lval = 0.0
            for (k,j) in enumerate(node.children)
                # ensure exp in only computed on nonpositive arguments (avoid overflow)
                lval += exp(values[j]-m)*node.weights[k]
            end
            # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf
            values[i] = isfinite(lval) ? m + log(lval) : -Inf
        else # is a leaf node
            values[i] = logpdf(node, x[node.scope])
        end
    end
    @inbounds return values[1]
end

"""
    plogpdf!(values,spn,layers,x)

Evaluates the sum-product network `spn` in log domain at configuration `x` using the scheduling in `nlayers` as obtained by the method `layers(spn)`. Stores values of each node in the vector `values`.

# Parameters

- `values`: vector to cache node values
- `spn`: the sum product network
- `nlayers`: Vector of vector of node indices determinig the layers of the `spn`; each node in a layer is computed based on values of nodes in smaller layers.
- `x`: Vector containing assignment

"""
function plogpdf!(values::AbstractVector{Float64}, spn::SumProductNetwork, nlayers, x::AbstractVector{<:Real})::Float64
    # visit layers from last (leaves) to first (root)
    @inbounds for l in length(nlayers):-1:1
        # parallelize computations within layer
        Threads.@threads for i in nlayers[l]
            node = spn[i]
            if isprod(node)
                lval = 0.0
                for j in node.children
                    lval += values[j]
                end
                values[i] = isfinite(lval) ? lval : -Inf
            elseif issum(node)
                # log-sum-exp trick to improve numerical stability (assumes weights are normalized)
                m = -Inf
                # get maximum incoming value
                for j in node.children
                    m = values[j] > m ? values[j] : m
                end
                lval = 0.0
                for (k,j) in enumerate(node.children)
                    # ensure exp in only computed on nonpositive arguments (avoid overflow)
                    lval += exp(values[j]-m)*node.weights[k]
                end
                # if something went wrong (e.g. incoming value is NaN or Inf) return -Inf, otherwise
                # return maximum plus lval (adding m ensures signficant digits are numerically precise)
                values[i] = isfinite(lval) ? m+log(lval) : -Inf
            else # is a leaf node
                values[i] = logpdf(node, x[node.scope])
            end
        end
    end
    @inbounds return values[1]
end

"""
    plogpdf(spn, x)

Parallelized version of `logpdf(spn, x)`. Set `JULIA_NUM_THREADS` > 1 to make this effective.
"""
function plogpdf(spn::SumProductNetwork, x::AbstractVector{<:Real})::Float64
    values = Array{Float64}(undef,length(spn))
    return plogpdf!(values,spn,layers(spn),x)
end


""" 
    sample(weights)::UInt

Sample integer with probability proportional to given `weights`. 

# Parameters

- `weights`: vector of nonnegative reals.
"""
function sample(weights)::UInt
    Z = sum(weights)
    u = rand()
    cum = 0.0
    for i = 1:length(weights)
        @inbounds cum += weights[i]/Z
        if u < cum
            return i
        end
    end
end

"""
    rand(n::IndicatorFunction)
    rand(n::CategoricalDistribution)
    rand(n::GaussianDistribution)

Sample values from sum-product network leaves.
"""
@inline Base.rand(n::IndicatorFunction) = n.value
@inline Base.rand(n::CategoricalDistribution) = sample(n.values)
@inline Base.rand(n::GaussianDistribution) = n.mean + sqrt(n.variance)*randn()

"""
    rand(spn)

Returns a sample of values of the variables generated according
to the probability defined by the network `spn`. Stores the sample
as a vector of values

# Example

```julia
julia> rand(spn)
[2, 1]
```
"""
function Base.rand(spn::SumProductNetwork)
    if length(scope(spn)) > 0
        numvars = length(scope(spn))
    else
        numvars = length(union(n.scope for n in leaves(spn)))
    end
    a = Vector{Float64}(undef, numvars)
    # sample induced tree
    queue = [1]
    while !isempty(queue)
        n = popfirst!(queue)
        if issum(spn[n])
            # sample one child to visit
            c = sample(spn[n].weights)  
            push!(queue, spn[n].children[c])
        elseif isprod(spn[n])
            # visit every child
            append!(queue, children(spn,n))
        else
            # draw value from node distribution
            a[ spn[n].scope ] = rand(spn[n])      
        end
    end
    return a
end

"""
  rand(spn::SumProductNetwork, N::Integer)

Returns a matrix of samples generated according to the probability
defined by the network `spn`. 
"""
function Base.rand(spn::SumProductNetwork, N::Integer)
    if length(scope(spn)) > 0
        numvars = length(scope(spn))
    else
        numvars = length(union(n.scope for n in leaves(spn)))
    end
    Sample = Array{Float64}(undef,N,numvars)
    # get sample
    for i=1:N
        queue = [1]
        while length(queue) > 0
            n = popfirst!(queue)
            if issum(spn[n])
                # sample one child to visit
                c = sample(spn[n].weights) # sparse array to vector inserts 0 at first coordinate
                push!(queue, spn[n].children[c])
            elseif isprod(spn[n])
                # visit every child
                append!(queue, children(spn,n))
            else
                # draw value from distribution
                Sample[ i, spn[n].scope ] = rand(spn[n])         
            end 
        end
    end
    return Sample
end

"""
    ncircuits(spn::SumProductNetwork)

Counts the number of induced circuits of the sum-product network `spn`.
"""
ncircuits(spn::SumProductNetwork) = ncircuits!(Array{Int}(undef,length(spn)),spn)
"""
    ncircuits!(values,spn)

Counts the number of induced circuits of the sum-product network `spn`, caching intermediate values.
"""
function ncircuits!(values::AbstractVector{Int}, spn::SumProductNetwork)
    @assert length(values) == length(spn)
    # traverse nodes in reverse topological order (bottom-up)
    for i in length(spn):-1:1
        @inbounds node = spn[i]
        if isa(node,ProductNode)
            @inbounds values[i] = mapreduce(j -> values[j], *, node.children)
        elseif isa(node,SumNode)
            @inbounds values[i] = sum(values[node.children])
        else # is a leaf node
            @inbounds values[i] = 1
        end
    end
    @inbounds values[1]
end

"""
    NLL(spn::SumProductNetwork,data::AbstractMatrix{<:Real})

Computes the average negative loglikelihood of a dataset `data` assigned by spn.
"""
NLL(spn::SumProductNetwork,data::AbstractMatrix{<:Real}) = -logpdf(spn,data)/size(data,1)

"""
    MAE(S1::SumProductNetwork,S2::SumProductNetwork,data::AbstractMatrix{<:Real})

Computes the Mean Absolute Error of sum-product networks `S1` and `S2` on given `data`. 
"""
MAE(S1::SumProductNetwork,S2::SumProductNetwork,data::AbstractMatrix{<:Real}) = sum( abs(logpdf(S1,view(data,i,:))-logpdf(S2,view(data,i,:))) for i=1:size(data,1))/size(data,1)
