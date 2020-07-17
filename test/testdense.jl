@info "## Dense Sum-Product Networks"
@testset "Dense SPNs" begin

    using DataFrames, CSV
    
    width,height = 16,16

    @info "Reading and processing dataset..."
    # Load data set
    @time dataset = CSV.read("../data/zip.test", 
         header=["digit"; collect(map(string, 1:width*height))], # columns names
         copycols=true # so that we can manipulate columns
    );
    nine = dataset[dataset.digit .== 9, 2:257];
    # count instances of digit 9 in training set
    println("Dataset has ", size(nine, 1), " instances.")
    # binarize data
    bin_nine = copy(nine)
    for i=1:(width*height)
        bin_nine[nine[i] .<= -0.5, i] .= 1
        idx = (nine[i] .> -0.5) .& (nine[i] .<= 0.5)
        bin_nine[idx, i] .= 2
        bin_nine[nine[i] .> 0.5, i] .= 3
    end
    # threshold = 0.0
    # for i=1:256
    #     bin_nine[nine[i] .<= threshold, i] .= 1
    #     bin_nine[nine[i] .> threshold, i] .= 2
    # end
    #println(first(bin_nine,5))
    data = convert(Matrix, bin_nine); 
    # @info "done"

    # build Dense SPN
    @info "Creating binary dense network..."
    @time S = buildDenseSPN(1,width,1,height,3)
    #@time S = learnspn(data,repeat([3],width*height))
    #@time S = buildDenseSPN(1,width,1,height,0)
    # @info "done." S
    @test length(S) == size(S._backward,1) + length(leaves(S))
    computed = falses(length(S))
    for i in Iterators.Reverse(1:length(S))
        computed[i] = true
        if isa(S[i],SumNode) || isa(S[i],ProductNode)
            if i > size(S._backward,1)
                @error "Inner node is not in backward" i size(S._backward,1)
            end
            for j in children(S,i)
                if computed[j] == false
                    @error "Parent computed before child" i j Type{S[i]} Type{S[j]}
                    error("Invalid spn")
                end
            end
        end
    end

    # @info "Printing graph structure to file..."
    # io = IOBuffer()
    # todot(io,S)
    # open("../examples/densespn.dot", "w") do f
    #     write(f, String(take!(io)))
    # end
    # println("done! Call dot -Tpdf examples/densespn.dot -o examples/densespn.pdf to generate image")
    
    @info "Learning parameters..."
    learner = EMParamLearner()
    learner.tolerance = 1e-3
    learner.minimumvariance = 0.1
    #println("Initializing weights...")
    #@time initialize(S) # randomly initialize weights
    # Now learn network parameters from data
    while !converged(learner) && learner.steps < 10
        @time SumProductNetworks.step(learner,S,data)
        println("It: $(learner.steps) \t NLL: $(-learner.score)")
        if !isfinite(learner.score)
            break
        end
    end
    # @info "done"


    # run inference
    @info "Testing if log evaluation is consistent..."
    @time logpdf(S,view(data,2,:))
    @test logpdf(S,data[1,:]) == logpdf(S,view(data,1,:))
    # @info "done."

    @info "Sampling from network..."
    @time x = rand(S)
    # @info "done."


    #r1 = RegionNode(1,8,2,6);
    #@test hash(r1) == 2*((1+8)+1*8)+(2+6)+2*6 + ((9+8)*(9+8+8+12)) 
    #r2 = RegionNode(2,6,1,8);
    #@test hash(r2) == 2*((2+6)+2*6)+(1+8)+1*8 + ((8+12)*(8+12+9+8)) 
    #@test hash(r1) != hash(r2)
    #p = PartitionNode([],1,8,2,6)
    #@test hash(r1) == hash(p)
    #@test isequal(r1,p)
    #@test r1 != p

end # END of test set
