# Implements node data types and associated helper functions

"""
Node Data Structures

Implement a labeled sparse matrix.
"""
abstract type Node end
struct SumNode <:Node
    children::Vector{UInt}
    weights::Vector{Float64}
    # SumNode() = new(Vector{UInt}(),Vector{Float64}())
    # SumNode(children::Vector{<:Integer},weights::Vector{Float64}) = new(children,weights)
end
struct ProductNode <: Node
    children::Vector{UInt}
    #ProductNode() = new(Vector{UInt}())
    #ProductNode(children::Vector{<:Integer}) = new(children)
    #ProductNode(children) = new(children)
end
abstract type LeafNode <: Node end
"""
Indicator Function Node. Tolerance sets a maximum discrepancy when evaluating the node at a given value. Its default value is 1e-6.
"""
struct IndicatorFunction <: LeafNode
    scope::UInt
    value::Float64
    tolerance::Float64
    IndicatorFunction(scope::Integer,value::Float64) = new(scope,value,1e-6)
    IndicatorFunction(scope::Integer,value::Integer) = new(scope,Float64(value),1e-6)
end
"""
Univariate Categorical Distribution Node
"""
struct CategoricalDistribution <: LeafNode
    scope::UInt
    values::Vector{Float64}
end
"""
Univariate Gaussian Distribution Node
"""
mutable struct GaussianDistribution <: LeafNode
    scope::UInt
    mean::Float64
    variance::Float64
end

# LeafNode = Union{IndicatorFunction,CategoricalDistribution,GaussianDistribution}

"""
    IndicatorFunction(x::Vector{<:Real})::Float64

Evaluates indicator function at given configuration x.
"""
function (n::IndicatorFunction)(x::AbstractVector{<:Real})::Float64
    return isnan(x[n.scope]) ? 1.0 : n.value ≈ x[n.scope] ? 1.0 : 0.0
end
"""
Evaluates categorical distribution at given configuration
"""
function (n::CategoricalDistribution)(x::AbstractVector{<:Real})::Float64
    return isnan(x[n.scope]) ? 1.0 : n.values[Int(x[n.scope])]
end
"""
Evaluates Gaussian distribution at given configuration
"""
function (n::GaussianDistribution)(x::AbstractVector{<:Real})::Float64
    return isnan(x[n.scope]) ? 1.0 : exp(-(x[n.scope]-n.mean)^2/(2*n.variance))/sqrt(2*π*n.variance)
end

"Is this a leaf node?"
@inline isleaf(n::Node) = isa(n,LeafNode)
"Is this a sum node?"
@inline issum(n::Node) = isa(n,SumNode)
"Is this a product node?"
@inline isprod(n::Node) = isa(n,ProductNode)

"""
    logpdf(node,value)

Evaluates leaf `node` at the given `value` in log domain.
"""
@inline logpdf(n::IndicatorFunction, value::Integer) = isnan(value) ? 0.0 : value == Int(n.value) ? 0.0 : -Inf
@inline logpdf(n::IndicatorFunction, value::Float64) = isnan(value) ? 0.0 : abs(value - n.value) < n.tolerance  ? 0.0 : -Inf
@inline logpdf(n::CategoricalDistribution, value::Integer) = @inbounds log(n.values[value])
@inline logpdf(n::CategoricalDistribution, value::Float64) = isnan(value) ? 0.0 : logpdf(n,Int(value))
@inline logpdf(n::GaussianDistribution, value::Float64)::Float64 = @inbounds isnan(value) ? 0.0 : (-(value-n.mean)^2/(2*n.variance)) - log(2*π*n.variance)/2
"""
    maximum(node)

Returns the maximum value of the distribution
"""
@inline Base.maximum(n::IndicatorFunction) = 1.0
@inline Base.maximum(n::CategoricalDistribution) = Base.maximum(n.values)
@inline Base.maximum(n::GaussianDistribution) =  1/sqrt(2*π*n.variance)
"""
    argmax(node)

Returns the value at which the distribution is maximum
"""
@inline Base.argmax(n::IndicatorFunction) = n.value
@inline Base.argmax(n::CategoricalDistribution) = Base.argmax(n.values)
@inline Base.argmax(n::GaussianDistribution) =  n.mean

"""
    scope(node)

Returns the scope of a leaf node
"""
scope(n::LeafNode) = n.scope
