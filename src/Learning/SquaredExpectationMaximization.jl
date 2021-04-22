# Parameter learning by Accelerated Expectation-Maximimzation (SQUAREM)
# RAVI VARADHAN & CHRISTOPHE ROLAND, Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm, J. Scand J Statist 2008
# Yu Du & Ravi Varadhan, SQUAREM: An R Package for Off-the-Shelf Acceleration of EM, MM and Other EM-Like Monotone Algorithms, J. Statistical Software, 2020.


"""
Learn weights using the Expectation Maximization algorithm. 
"""
mutable struct SQUAREM <: ParameterLearner
spn::SumProductNetwork
layers::Vector{Vector{Int}}
cache1::SumProductNetwork
cache2::SumProductNetwork
cache3::SumProductNetwork
cache4::SumProductNetwork
diff::Vector{Float64} # to store derivatives
values::Vector{Float64} # to store logprobabilities
# dataset::AbstractMatrix
score::Float64     # score (loglikelihood)
prevscore::Float64 # for checking convergence
tolerance::Float64 # tolerance for convergence criterion
steps::Integer   # number of learning steps (epochs)
minimumvariance::Float64 # minimum variance for Gaussian leaves
SQUAREM(spn::SumProductNetwork) = new(spn, layers(spn), deepcopy(spn), deepcopy(spn), deepcopy(spn), deepcopy(spn), Array{Float64}(undef,length(spn)), Array{Float64}(undef,length(spn)), NaN, NaN, 1e-3, 0, 0.5)
#TODO: use sparse tensor of weight updates
end

"""
Random initialization of weights
"""
function initialize(learner::SQUAREM) #spn::SumProductNetwork)
    spn = learner.spn
    sumnodes = filter(i -> isa(spn[i], SumNode), 1:length(spn))
    for i in sumnodes
        #@inbounds ch = spn[i].children  # children(spn,i)        
        @inbounds Random.rand!(spn[i].weights)
        @inbounds spn[i].weights ./= sum(spn[i].weights) 
        @assert sum(spn[i].weights) ≈ 1.0 "Unnormalized weight vector at node $i: $(sum(spn[i].weights)) | $(spn[i].weights)"
    end    
    gaussiannodes = filter(i -> isa(spn[i],GaussianDistribution), 1:length(spn))
    for i in gaussiannodes
        @inbounds spn[i].mean = rand()
        @inbounds spn[i].variance = 1.0
    end
end

""" 
Verifies if algorithm has converged.

Convergence is defined as absolute difference of score. 
Requires at least 2 steps of optimization.
"""
converged(learner::SQUAREM) = learner.steps < 2 ? false : abs(learner.score - learner.prevscore) < learner.tolerance

# TODO: take filename os CSV File object as input and iterate over file to decreae memory footprint
"""
Improvement step for learning weights using the Squared Iterative Expectation Maximization algorithm. Returns improvement in negative loglikelihood.

spn[i].weights[j] = spn[i].weights[k] * backpropagate(spn)[i]/sum(spn[i].weights[k] * backpropagate(spn)[i] for k=1:length(spn[i].children))

## Arguments

- `learner`: SQUAREM struct
- `data`: Data Matrix
- `smoothing`: weight smoothing factor (= pseudo expected count) [default: 0.1]
- `minimumvariance`: minimum variance for Gaussian leaves [default: learner.minimumvariance]
"""
function update(learner::SQUAREM, Data::AbstractMatrix, smoothing::Float64 = 0.1, minimumvariance::Float64 = learner.minimumvariance)
        
    numrows, numcols = size(Data)

    θ_0 = learner.spn
    θ_1 = learner.cache1
    θ_2 = learner.cache2
    r = learner.cache3
    v = learner.cache4
    # smooth out estaimtors to avoid degenerate probabilities
    sumnodes = filter(i -> isa(learner.spn[i], SumNode), 1:length(learner.spn))
    diff = learner.diff
    values = learner.values
    # Compute theta1 = EM_Update(theta0)
    for t = 1:numrows
        datum = view(Data, t, :)
        lv = plogpdf!(values, θ_0, learner.layers, datum) # parallelized version
        @assert isfinite(lv) "1. logvalue of datum $t is not finite: $lv"
        backpropagate!(diff, θ_0, values) # backpropagate derivatives
        Threads.@threads for i in sumnodes # update each node in parallel
            @inbounds for (k,j) in enumerate(learner.spn[i].children)
                if isfinite(values[j])
                    δ = θ_0[i].weights[k]*diff[i]*exp(values[j]-lv) # improvement
                    @assert isfinite(δ) "1. improvement to weight ($i,$j):$(θ_0[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
                else
                    δ = 0.0
                end
                θ_1[i].weights[k] = ((t-1)/t)*θ_1[i].weights[k] + δ/t # running average for improved precision
                @assert θ_1[i].weights[k] ≥ 0
            end
        end
    end
    @inbounds Threads.@threads for i in sumnodes
        θ_1[i].weights .+= smoothing/length(θ_1[i].weights) # smoothing factor to prevent degenerate probabilities
        θ_1[i].weights ./= sum(θ_1[i].weights) 
        @assert sum(θ_1[i].weights) ≈ 1.0 "1. Unnormalized weight vector at node $i: $(sum(θ_1[i].weights)) | $(θ_1[i].weights)"
    end
    # Compute theta2 = EM_Update(theta1)
    for t = 1:numrows
        datum = view(Data, t, :)
        lv = plogpdf!(values, θ_1, learner.layers, datum) # parallelized version
        @assert isfinite(lv) "2. logvalue of datum $t is not finite: $lv"
        backpropagate!(diff, θ_1, values) # backpropagate derivatives
        Threads.@threads for i in sumnodes # update each node in parallel
            @inbounds for (k,j) in enumerate(learner.spn[i].children)
                if isfinite(values[j])
                    δ = θ_1[i].weights[k]*diff[i]*exp(values[j]-lv) # improvement
                    @assert isfinite(δ) "2. improvement to weight ($i,$j):$(θ_1[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
                else
                    δ = 0.0
                end
                θ_2[i].weights[k] = ((t-1)/t)*θ_2[i].weights[k] + δ/t
                @assert θ_2[i].weights[k] ≥ 0
            end
        end
    end
    @inbounds Threads.@threads for i in sumnodes
        θ_2[i].weights .+= smoothing/length(θ_2[i].weights) # smoothing factor to prevent degenerate probabilities
        θ_2[i].weights ./= sum(θ_2[i].weights) 
        @assert sum(θ_2[i].weights) ≈ 1.0 "2. Unnormalized weight vector at node $i: $(sum(θ_2[i].weights)) | $(θ_2[i].weights)"
    end    
    # Compute r, v, |r| and |v|
    r_norm, v_norm = 0.0, 0.0
    @inbounds Threads.@threads for i in sumnodes
        # r[i].weights .= θ_1[i].weights .- θ_0[i].weights
        # v[i].weights .= θ_2[i].weights .- θ_1[i].weights .- r[i].weights
        for k in 1:length(r[i].weights)
            r[i].weights[k] = θ_1[i].weights[k] - θ_0[i].weights[k]
            v[i].weights[k] = θ_2[i].weights[k] - θ_1[i].weights[k] - r[i].weights[k]
            r_norm += r[i].weights[k] * r[i].weights[k]
            v_norm += v[i].weights[k] * v[i].weights[k]
        end
        # r_norm += sum(r[i].weights .* r[i].weights)
        # v_norm += sum(v[i].weights .* v[i].weights)
    end 
    # steplength   
    α = -max(sqrt(r_norm)/sqrt(v_norm),1)
    #println("α: $α")
    # Compute θ' (reuse θ_1 for that matter)
    @inbounds Threads.@threads for i in sumnodes
        # θ' = θ0 - 2αr + α^2v
        θ_1[i].weights .= θ_0[i].weights 
        θ_1[i].weights .-= ((2*α).*r[i].weights)
        θ_1[i].weights .+= ((α*α).*v[i].weights) 
        θ_1[i].weights .+ smoothing/length(θ_1[i].weights) # add term to prevent negative weights due to numerical imprecision
        θ_1[i].weights ./= sum(θ_1[i].weights)
        @assert sum(θ_1[i].weights) ≈ 1.0 "3. Unnormalized weight vector at node $i: $(sum(θ_1[i].weights)) | $(θ_1[i].weights)"
        for w in θ_1[i].weights
            @assert w ≥ 0 "Negative weight at node $i: $(θ_1[i].weights)"
        end
    end    
    # Final EM Update: θ_0 = EM_Update(θ')
    score = 0.0 # data loglikelihood
    for t = 1:numrows
        datum = view(Data, t, :)
        lv = plogpdf!(values, θ_1, learner.layers, datum) # parallelized version
        @assert isfinite(lv) "4. logvalue of datum $t is not finite: $lv"
        score += lv
        backpropagate!(diff, θ_1, values) # backpropagate derivatives
        Threads.@threads for i in sumnodes # update each node in parallel
            @inbounds for (k,j) in enumerate(learner.spn[i].children)
                if isfinite(values[j])
                    δ = θ_1[i].weights[k]*diff[i]*exp(values[j]-lv) # improvement
                    @assert isfinite(δ) "4. improvement to weight ($i,$j):$(θ_1[i].weights[k]) is not finite: $δ, $(diff[i]), $(values[j]), $(exp(values[j]-lv))"
                else
                    δ = 0.0
                end
                θ_0[i].weights[k] = ((t-1)/t)*θ_0[i].weights[k] + δ/t
                @assert θ_0[i].weights[k] ≥ 0
            end
        end
    end
    @inbounds Threads.@threads for i in sumnodes
        θ_0[i].weights .+= smoothing/length(θ_0[i].weights) # smoothing factor to prevent degenerate probabilities
        θ_0[i].weights ./= sum(θ_0[i].weights) 
        @assert sum(θ_0[i].weights) ≈ 1.0 "4. Unnormalized weight vector at node $i: $(sum(θ_0[i].weights)) | $(θ_0[i].weights)"
    end
    learner.steps += 1
    learner.prevscore = learner.score
    learner.score = -score/numrows
    return learner.prevscore-learner.score, α
end

