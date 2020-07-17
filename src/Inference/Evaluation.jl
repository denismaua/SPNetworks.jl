# Routines for SPN evaluation and sampling

export
    logpdf, 
    logpdf!,
    rand,
    NLL,
    MAE


"""
  (spn::SumProductNetwork)(x::AbstractVector{<:Real})::Float64

Evaluates the sum-product network at a given instantiation.

`x` is vector of values of variables (integers or reals). Summed-out variables are represented as NaNs

# Examples

To compute the probability of P(b=1) using a spn S, use
```julia
julia> S([NaN, 2]
0.3
```
"""
function (spn::SumProductNetwork)(x::AbstractVector{<:Real})::Float64
    return exp(logpdf(spn,x))
end


"""
    logpdf(spn, X)

Returns the sums of the log-probabilities of instances x in X.

`X` is matrix of values of variables (integers or reals). Summed-out variables are represented as NaNs.
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
    logpdf(spn, x)

Evaluates the sum-product network `spn` in log domain at configuration `x`.

`x` is vector of values of variables (integers or reals). Summed-out variables are represented as NaNs

# Examples

To compute the probability of ``P(b=1)`` using a spn `S`, use
```julia
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

Evaluates the sum-product network `spn` in log domain at configuration `x` and stores values of each node in the vector values.
"""
function logpdf!(values::AbstractVector{Float64}, spn::SumProductNetwork, x::AbstractVector{<:Real})::Float64
    @assert length(values) == length(spn)
    # traverse nodes in reverse topological order (bottom-up)
    for i in length(spn):-1:1
        @inbounds node = spn[i]
        if isa(node,ProductNode)
            lval = 0.0
            for j in node.children
                @inbounds lval += values[j]
            end
            @inbounds values[i] = isfinite(lval) ? lval : -Inf
        elseif isa(node,SumNode)
            # log trick to improve numerical stability
            m = -Inf
            for j in node.children
                @inbounds m = values[j] > m ? values[j] : m
            end
            lval = 0.0
            for (k,j) in enumerate(node.children)
                @inbounds lval += exp(values[j]-m)*node.weights[k]
            end
            @inbounds values[i] = isfinite(lval) ? log(lval)+m : -Inf
        else # is a leaf node
            @inbounds values[i] = logpdf(node,x[node.scope])
        end
    end
    @inbounds return values[1]
end

""" Sample integer with probability proportional to given weights. """
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
Sample values from leaves.
"""
@inline Base.rand(n::IndicatorFunction) = n.value
@inline Base.rand(n::CategoricalDistribution) = sample(n.values)

"""
    rand(spn)

Returns a sample of values of the variables generated according
to the probability defined by the network `spn`. Stores the sample
as a vector of values

# Example

```julia
julia> println(rand(spn))
[2, 1]
```
"""
function Base.rand(spn::SumProductNetwork)::Vector{Float64}
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
        if isa(spn[n],SumNode)
            # sample one child to visit
            c = sample(spn[n].weights)  
            push!(queue, spn[n].children[c])
        elseif isa(spn[n],ProductNode)
            # visit every child
            append!(queue, children(spn,n))
        elseif isa(spn[n],CategoricalDistribution)
            # draw value from distribution
            a[ spn[n].scope ] = rand(spn[n])         
        elseif isa(spn[n],GaussianDistribution)
            # draw from Gaussian distribution
            a[ spn[n].scope ] = spn[n].mean + sqrt(spn[n].variance)*randn()
        end
    end
    return a
end

"""
  rand(spn::SumProductNetwork, N::Integer)

Returns a matrix of samples generated according to the probability
defined by the network `spn`. 
"""
function Base.rand(spn::SumProductNetwork, N::Integer)::Matrix
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
            if isa(spn[n],SumNode)
                # sample one child to visit
                c = sample(spn[n].weights) # sparse array to vector inserts 0 at first coordinate
                push!(queue, spn[n].children[c])
            elseif isa(spn[n],ProductNode)
                # visit every child
                append!(queue, children(spn,n))
            elseif isa(spn[n],CategoricalDistribution)
                # draw value from distribution
                Sample[ i, spn[n].scope ] = rand(spn[n])         
            elseif isa(spn[n],GaussianDistribution)
                Sample[ i, spn[n].scope ] = spn[n].mean + sqrt(spn[n].variance)*randn()
            end 
        end
    end
    return Sample
end


"""
Computes the average negative loglikelihood of a dataset data assigned by spn.
"""
NLL(spn::SumProductNetwork,data::AbstractMatrix{<:Real}) = -logpdf(spn,data)/size(data,1)

"""
Computes the Mean Absolute Error of two SPNs on given data
"""
MAE(S1::SumProductNetwork,S2::SumProductNetwork,data::AbstractMatrix{<:Real}) = sum( abs(logpdf(S1,view(data,i,:))-logpdf(S2,view(data,i,:))) for i=1:size(data,1))/size(data,1)
