# Compares EM variants on given network structure
# Example of commaind line call: 
#     export JULIA_NUM_THREADS=4; julia --color=yes em.jl assets/nltcs.spn assets/nltcs.train.csv assets/nltcs.valid.csv 100 
#using DataFrames, CSV
using DelimitedFiles # much faster than above
using Random
using SPNetworks
import SPNetworks: NLL
import SPNetworks.ParameterLearning: EMParamLearner, initialize, converged, update, backpropagate

if length(ARGS) < 3
    println("Usage: julia --color=yes em.jl spn_filename train_filename validation_filename [batchsize]")
    exit()
end
spn_filename = normpath(ARGS[1])
train_filename = normpath(ARGS[2])
valid_filename = normpath(ARGS[3])
batchsize = 100 # default
if length(ARGS) >= 3
    batchsize = parse(Int, ARGS[4])
end
println("Maximum batchsize: $batchsize")
println("Threads: ", Threads.nthreads())

println("SPN: $spn_filename")
spn = SumProductNetwork(spn_filename)
@show summary(spn)
#SPNetworks.save(spn,normpath("$train_filename"))
nvars = length(scope(spn))
println("Training data: $train_filename")
# Load datasets
# @time tdata = convert(Matrix{Float64},
#             DataFrame(
#                 CSV.File(train_filename, 
#                     header=collect(1:nvars) # columns names
#                 ) 
#             )
#         ) .+ 1;
@time tdata = readdlm(train_filename, ',', Float64) .+ 1;
@show summary(tdata)
println("Validation data: $valid_filename")
@time vdata = readdlm(valid_filename, ',', Float64) .+ 1;
# @time vdata = convert(Matrix{Float64},
#             DataFrame(
#                 CSV.File(valid_filename, 
#                     header=collect(1:nvars) # columns names
#                 ) 
#             )
#         ) .+ 1;
@show summary(vdata)
# initialize EM learner
learner = EMParamLearner(spn)
Random.seed!(3)
#println("It: $(learner.steps) \t NLL: $(learner.score) \t held-out NLL: $(NLL(spn, vdata))")
#initialize(learner) # generate random weights for each sum node
indices = shuffle!(Vector(1:size(tdata,1)))
# Running Expectation Maximization
while !converged(learner) && learner.steps < 3 # 11
    sid = rand(1:(length(indices)-batchsize))
    batch = view(tdata, indices[sid:(sid+batchsize-1)], :) # extract minibatch sample
    if learner.steps % 2 == 0          
        tnll = NLL(spn, vdata)
        @time update(learner, batch, 0.9^learner.steps)
        println("It: $(learner.steps) \t NLL: $(learner.score) \t held-out NLL: $tnll")
    else
        @time update(learner, batch, 0.9^learner.steps)
        println("It: $(learner.steps) \t NLL: $(learner.score)")
    end
end
