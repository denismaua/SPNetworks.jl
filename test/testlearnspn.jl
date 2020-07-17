# Test Parameter Learning routines

# @testset "SPN structure learning: LearnSPN" begin

#     println()

#     @info " Generating true model..."
#     # First create SPN to generate data from
#     A = Vector([
#         SumNode(),     # 1
#         ProductNode(), # 2
#         ProductNode(), # 3
#         ProductNode(), # 4
#         CategoricalDistribution(1, [0.6,0.4]), # 5
#         CategoricalDistribution(1, [0.1,0.9]), # 6
#         CategoricalDistribution(2, [0.3,0.7]), # 7
#         CategoricalDistribution(2, [0.8,0.2])  # 8
#     ])
#     B = [
#         [2, 3, 4], # 1 Sum P(A, B) 0.2,0.5,0.3
#         [5, 7],    # 2 Prod P(A, B)=P(A)*P(B)
#         [5, 8],    # 3 Prod P(A, B)=P(A)*P(B)
#         [6, 8],    # 4 Prod P(A, B)=P(A)*P(B)
#     ]
#     C = sparse([2,   3,     4],   
#                [1,   1,     1],
#                [0.2, 0.5, 0.3])
    
#     SPN = SumProductNetwork{Int64}(A, B, C)
#     @info " Generating data set..."
#     N = 3000
#     println("N = $N")
#     data = rand(SPN,N)

#     @info " Learning SPN structure from data..."
#     @time SPN2 = learnspn(data,[2,2])
#     println(SPN2)

#     # @info "## Printing graph structure"
#     # io = IOBuffer()
#     # todot(io,SPN2)
#     # open("spn.dot", "w") do f
#     #     write(f, String(take!(io)))
#     # end
#     # println("Call 'dot -Tpdf spn.dot -o spn.pdf' to generate image")

#     #@test MAE(SPN,SPN2,data) < 0.05

#     println()
#     Z = 0.0
#     for a in 1:2, b in 1:2
#         ref = SPN([a,b])
#         est = SPN2([a,b])
#         println("S($a,$b) = $ref ≈ ", est)
#         Z += est
#     end
#     @test Z ≈ 1.0
#     println()

# end

@testset "LearnSPN: NLTCS" begin

    using DataFrames, CSV


    @info "Reading and processing dataset..."
    # Load data set
    @time dataset = CSV.read("../data/nltcs.train.csv", 
                             header=collect(1:16) # columns names
    );

    println("Dataset has $(size(dataset,1)) instances, $(size(dataset,2)) columns.")

    data = convert(Matrix, dataset) .+ 1
    # @info "done"

    # build Dense SPN
    @info "Learning structure..."
    @time S = learnspn(data,repeat([2],16,),100)
    @info "done." S
    #save(S,"../examples/nltcs.learnspn.notuning.spn")

    @info "Evaluating learned model..."
    @time testset = CSV.read("../data/nltcs.test.csv", 
                             header=collect(1:16) # columns names
    );
    println("Testset has $(size(testset,1)) instances, $(size(testset,2)) columns.")
    test = convert(Matrix, testset) .+ 1
    println("Avg Training Negative LL: ", NLL(S,data)) 
    println("Avg Testset Negative LL: ", NLL(S,test)) 


    @info "Tuning parameters..."
    learner = EMParamLearner()
    learner.tolerance = 1e-6
    #println("Initializing weights...")
    #initialize(S) # randomly initialize weights
    # Now learn network parameters from data
    while !converged(learner) && learner.steps < 500
        SumProductNetworks.step(learner,S,data)
        println("It: $(learner.steps) \t ANLL: $(-learner.score) \t ANLL: $(NLL(S,test))")
        if !isfinite(learner.score)
            break
        end
    end
    println("Avg Training Negative LL: ", NLL(S,data)) 
    println("Avg Testset Negative LL: ", NLL(S,test)) 

    #save(S,"../examples/nltcs.learnspn.em.spn")

end


# @testset "LearnSPN: ZIP dataset" begin

#     using DataFrames, CSV
    
#     width,height = 16,16

#     @info "Reading and processing dataset..."
#     # Load data set
#     @time dataset = CSV.read("../data/zip.test", 
#          header=["digit"; collect(map(string, 1:width*height))], # columns names
#          copycols=true # so that we can manipulate columns
#     );
#     nine = dataset[dataset.digit .== 9, 2:257];
#     # count instances of digit 9 in training set
#     println("Dataset has $(size(nine,1)) instances, $(size(nine,2)) columns.")
#     # binarize data
#     bin_nine = copy(nine)
#     for i=1:(width*height)
#         bin_nine[nine[i] .<= -0.5, i] .= 1
#         idx = (nine[i] .> -0.5) .& (nine[i] .<= 0.5)
#         bin_nine[idx, i] .= 2
#         bin_nine[nine[i] .> 0.5, i] .= 3
#     end
#     # threshold = 0.0
#     # for i=1:256
#     #     bin_nine[nine[i] .<= threshold, i] .= 1
#     #     bin_nine[nine[i] .> threshold, i] .= 2
#     # end
#     #println(first(bin_nine,5))
#     data = convert(Matrix, bin_nine); 
#     # @info "done"

#     # build Dense SPN
#     @info "Learning structure..."
#     @time S = learnspn(data,repeat([3],width*height))
#     @info "done." S
#     print("Cardinalities: ")
#     for node in leaves(S)
#         print("$(length(node.values)) ")
#     end
#     println()
#     computed = falses(length(S))
#     for i in Iterators.Reverse(1:length(S))
#         computed[i] = true
#         if isa(S[i],SumNode) || isa(S[i],ProductNode)
#             if i > size(S._backward,1)
#                 @error "Inner node is not in backward" i size(S._backward,1)
#             end
#             for j in children(S,i)
#                 if computed[j] == false
#                     @error "Parent computed before child" i j Type{S[i]} Type{S[j]}
#                     error("Invalid spn")
#                 end
#             end
#         end
#     end

#     # @info "Printing graph structure to file..."
#     # io = IOBuffer()
#     # todot(io,S)
#     # open("../examples/densespn.dot", "w") do f
#     #     write(f, String(take!(io)))
#     # end
#     # println("done! Call dot -Tpdf examples/densespn.dot -o examples/densespn.pdf to generate image")
    
#     @info "Learning parameters..."
#     learner = EMParamLearner()
#     learner.tolerance = 1e-3
#     learner.minimumvariance = 0.1
#     #println("Initializing weights...")
#     #@time initialize(S) # randomly initialize weights
#     # Now learn network parameters from data
#     while !converged(learner) && learner.steps < 10
#         SumProductNetworks.step(learner,S,data)
#         println("It: $(learner.steps) \t NLL: $(-learner.score)")
#         if !isfinite(learner.score)
#             break
#         end
#     end
#     # @info "done"


#     # run inference
#     @info "Testing if log evaluation is consistent..."
#     @time logpdf(S,view(data,2,:))
#     @test logpdf(S,data[1,:]) == logpdf(S,view(data,1,:))
#     # @info "done."

#     @info "Sampling from network..."
#     @time x = rand(S)
#     # @info "done."

# end # END of test set
