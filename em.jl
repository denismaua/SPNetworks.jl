# Compares EM variants on given network structure
# Example of commaind line call: 
#     julia --threads=4 --color=yes em.jl assets/nltcs.spn assets/nltcs.train.csv assets/nltcs.valid.csv 100 
#using DataFrames, CSV
using DelimitedFiles # much faster than above
using Random
using SPNetworks
import SPNetworks: NLL
import SPNetworks.ParameterLearning: SEM, initialize, converged, update

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
@time tdata = readdlm(train_filename, ',', Float64) .+ 1;
@show summary(tdata)
println("Validation data: $valid_filename")
@time vdata = readdlm(valid_filename, ',', Float64) .+ 1;
@show summary(vdata)
# initialize Stochastic EM learner
learner = SEM(spn)
Random.seed!(3)
#initialize(learner) # generate random weights for each sum node
avgnll = NLL(spn, tdata)
runnll = 0.0
println("It: $(learner.steps) \t train NLL: $avgnll \t held-out NLL: $(NLL(spn, vdata))")
indices = shuffle!(Vector(1:size(tdata,1)))
# Running Expectation Maximization
while !converged(learner) && learner.steps < 31
    global avgnll, runnll
    sid = rand(1:(length(indices)-batchsize))
    batch = view(tdata, indices[sid:(sid+batchsize-1)], :) # extract minibatch sample
    # if learner.steps % 2 == 0      
        η = max(0.95^learner.steps, 0.3) # learning rate
        @time update(learner, batch, η)
        testnll = NLL(spn, vdata)
        batchnll = NLL(spn, batch)
        # running average NLL
        avgnll *= (learner.steps-1)/learner.steps # discards initial NLL
        avgnll += batchnll/learner.steps
        runnll = (1-η)*runnll + η*batchnll
        println("It: $(learner.steps) \t avg NLL: $avgnll \t mov NLL: $runnll \t batch NLL: $batchnll \t held-out NLL: $testnll \t Learning rate: $η")
    # else
    #     @time update(learner, batch, 0.9^learner.steps)
    #     println("It: $(learner.steps) \t NLL: $(learner.score)")
    # end
end
