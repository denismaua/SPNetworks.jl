# Parameter Learning

module ParameterLearning

using SPNetworks
import SPNetworks: 
    Node, SumNode, ProductNode, LeafNode, CategoricalDistribution, IndicatorFunction, GaussianDistribution,
    isleaf, isprod, issum,
    logpdf!,
    vardims
import Random

"""
    Parameter Learning Algorithm
"""
abstract type ParameterLearner end

"""
    Computes the partial derivatives of the network output w.r.t. to each node at the current evaluation. Assumes network has been evaluated first in original domain.

    Returns a vector of derivatives.
"""
# function backpropagaten!(spn::SumProductNetwork)::Vector{Float64}
#     # Assumes network has been evaluted at some assignment
#     # Backpropagate derivatives
#     diff = zeros(Float64,length(spn._nodes)) 
#     diff[1] = 1.0
#     for (i,node) in enumerate(spn._nodes)
#         if isa(node, SumNode)
#             for j in spn._backward[i]
#                 diff[j] += spn._weights[j,i]*diff[i]
#             end
#         elseif isa(node, ProductNode)
#             for j in spn._backward[i]
#                 diff[j] += diff[i]*spn._values[i]/spn._values[j]
#             end
#         end
#     end
#     return diff
# end

# """
#  Computes derivatives for values propagated in log domain.
# Returns vector of derivatives.
# """
# function backpropagate(spn::SumProductNetwork)::Vector{Float64}
#     # Assumes network has been evaluted at some assignment using logpdf!
#     # Backpropagate derivatives
#     diff = zeros(Float64,length(spn)) 
#     values = similar(diff)
#     logpdf(spn,values)
#     backpropagate(spn,values,diff)
#     # diff[1] = 1.0
#     # for i = 1:length(spn)
#     #     if isa(spn[i], SumNode)
#     #         for j in children(spn,i)
#     #             diff[j] += spn._weights[j,i]*diff[i]
#     #         end
#     #     elseif isa(spn[i], ProductNode)
#     #         for j in children(spn,i) 
#     #             diff[j] += diff[i]*exp(spn._values[i]-spn._values[j])
#     #         end
#     #     end
#     # end
#     return diff
# end

"""
    backpropagate!(diff,spn,values)

 Computes derivatives for values propagated in log domain and stores results in given vector diff.
"""
function backpropagate!(diff::Vector{Float64},spn::SumProductNetwork,values::Vector{Float64})
    # Assumes network has been evaluted at some assignment using logpdf!
    # Backpropagate derivatives
    @assert length(diff) == length(spn) == length(values)
    fill!(diff,0.0)
    @inbounds diff[1] = 1.0
    for i = 1:length(spn)
        @inbounds node = spn[i]
        if issum(node)
            for (k,j) in enumerate(node.children)
               @inbounds diff[j] += node.weights[k]*diff[i]
            end
        elseif isprod(node)
            for j in node.children   
                if isfinite(values[j])
                    # @assert isfinite(exp(values[i]-values[j]))  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(exp(values[i]-values[j]))"
                    @inbounds diff[j] += diff[i]*exp(values[i]-values[j])
                else
                    δ = exp(sum(values[k] for k in node.children if k ≠ j))
                    # @assert isfinite(δ)  "contribution to derivative of ($i,$j) is not finite: $(values[i]), $(values[j]), $(δ)"
                    @inbounds diff[j] += diff[i]*δ
                end
            end
        end
    end
end

# compute log derivatives (not working!)
function logbackpropagate(spn::SumProductNetwork,values::Vector{Float64},diff::Vector{Float64})
    # Assumes network has been evaluted at some assignment using logpdf!
    # Backpropagate derivatives
    @assert length(diff) == length(spn) == length(values)
    # create storage for each computed value (message from parent to child)
    from = []
    to = []
    for i = 1:length(spn)
        if isa(spn[i], SumNode) || isa(spn[i], ProductNode)
            for j in children(spn,i)
                push!(from,i)
                push!(to,j)
            end
        end
    end
    cache = sparse(to,from,ones(Float64,length(to)))
    #fill!(diff,0.0)
    logdiff = zeros(Float64,length(diff))
    for i = 1:length(spn)
        if i == 1
            diff[i] == 1.0
        else
            cache_vals = nonzeros(cache[i,:]) # incoming arc values
            offset = maximum(cache_vals)
            logdiff[i] = offset + log(sum(exp.(cache_vals.-offset)))
            diff[i] = isfinite(logdiff[i]) ? exp(logdiff[i]) : 0.0
        end
        if isa(spn[i], SumNode)
            for j in children(spn,i)
                #@inbounds diff[j] += getweight(spn,i,j)*diff[i]
                cache[j,i] = logweight(spn,i,j) + logdiff[i]
            end
        elseif isa(spn[i], ProductNode)
            for j in children(spn,i)
                #@inbounds diff[j] += diff[i]*exp(values[i]-values[j])
                cache[j,i] = logdiff[i] + values[i]-values[j]
            end
        end
    end
end

"""
 Computes derivatives for given vector of values propagated in log domain.

Returns vector of derivatives.
"""
function backpropagate(spn::SumProductNetwork,values::Vector{Float64})::Vector{Float64}
    diff = Array{Float64}(undef,length(spn))
    backpropagate!(diff,spn,values)
    return diff
end

include("ExpectationMaximization.jl")

end