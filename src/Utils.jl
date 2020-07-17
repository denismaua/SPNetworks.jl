# Helper functions

"""
    expw = normexp!(logw,expw)

Compute expw .= exp.(logw)/sum(exp.(logw)). Uses the log-sum-exp trick to control for overflow in exp function.

Smoothing adds constant value to each value prior to normalization (useful to avoid zero probabilities).
"""
function normexp!(logw,expw,smoothing=0.0)
    offset = maximum(logw)
    expw .= exp.(logw .- offset) .+ smoothing
    s = sum(expw)
    expw .*= 1/s
end
# function logsumexp!(w,we)
#     offset = maximum(w)
#     we .= exp.(w .- offset)
#     s = sum(we)
#     w .-= log(s) + offset # this computes logw
#     we .*= 1/s
# end
# """
#     logΣexp, Σ = logsumexp!(p::WeightedParticles)
# Return log(∑exp(w)). Modifies the weight vector to `w = exp(w-offset)`
# Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow.

# References:
# https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
# https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
# """
# function logsumexp!(p::WeightedParticles)
#     N = length(p)
#     w = p.logweights
#     offset, maxind = findmax(w)
#     w .= exp.(w .- offset)
#     Σ = sum_all_but(w,maxind) # Σ = ∑wₑ-1
#     log1p(Σ) + offset, Σ+1
# end
# function sum_all_but(w,i)
#     w[i] -= 1
#     s = sum(w)
#     w[i] += 1
#     s
# end
