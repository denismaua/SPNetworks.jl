# Test Parameter Learning routines

@testset "SPN parameter learning" begin

    import SPNetworks: NLL
    import SPNetworks.ParameterLearning: EMParamLearner, initialize, converged, update, backpropagate

    @testset "Simple DAG SPN" begin

        # # First create SPN to generate data from
        # SPN = SumProductNetwork(
        #     [
        #         SumNode([2,3,4],[0.2,0.5,0.3]),        # 1
        #         ProductNode([5,7]),                    # 2
        #         ProductNode([5,8]),                    # 3
        #         ProductNode([6,8]),                    # 4
        #         CategoricalDistribution(1, [0.6,0.4]), # 5
        #         CategoricalDistribution(1, [0.1,0.9]), # 6
        #         CategoricalDistribution(2, [0.3,0.7]), # 7
        #         CategoricalDistribution(2, [0.8,0.2])  # 8
        #     ]
        # )    
        # # @info " Generating data set..."
        # N = 1000
        # data = rand(SPN,N)
            
            
        # Create SPN with indicator distributions at leaves
        SPN = SumProductNetwork(
            [
                SumNode([2,3,4],[0.2,0.5,0.3]),        # 1
                ProductNode([5,7]),                    # 2
                ProductNode([5,8]),                    # 3
                ProductNode([6,8]),                    # 4
                SumNode([9,10],[0.6,0.4]),             # 5
                SumNode([9,10],[0.1,0.9]),             # 6
                SumNode([11,12],[0.3,0.7]),            # 7
                SumNode([11,12],[0.8,0.2]),            # 8
                IndicatorFunction(1, 1.),              # 9
                IndicatorFunction(1, 2.),              # 10
                IndicatorFunction(2, 1.),              # 11
                IndicatorFunction(2, 2.)               # 12
            ]
        )        
        # Generate dataset
        N = 3000
        data = rand(SPN,N)

        learner = EMParamLearner(SPN)
        #initialize(learner)
        # @info "Running Expectation Maximization until convergence..."
        while !converged(learner) && learner.steps < 10
            update(learner, data)
            # println("It: $(learner.steps) \t NLL: $(learner.score)")
            # println("It: $(learner.steps) \t NLL: $(learner.score) \t MAE: $error")
        end
        # # Reasonable values
        # @test learner.score > -14000 && learner.score < -12000

        # println()
        Z = 0.0
        @testset "Evaluation at $a,$b" for a in 1:2, b in 1:2
            est = SPN([a,b])
            # empirical distribution
            emp = sum(1.0 for i=1:N if data[i,1] == a && data[i,2] == b)/N
            # println("S($a,$b) = $ref ≈ ", est)
            @test est ≈ emp atol=0.05
            # @test ref ≈ est atol=0.01
            Z += est
        end
        @test Z ≈ 1.0

    end
    @testset "Breast-Cancer SPN" begin
        # load larger SPN
        SPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/breast-cancer.spn"); offset = 1)
        #@show summary(SPN)
        # Generate dataset
        N = 3000
        data = rand(SPN, N)
        nll = NLL(SPN, data)
        learner = EMParamLearner(SPN)
        #initialize(learner)
        while !converged(learner) && learner.steps < 2
            update(learner, data)
            #println("It: $(learner.steps) \t NLL: $(learner.score)")
        end
        # # Training set NLL must be smaller than sampling distribution's training set NLL        
        @test learner.score < nll
    end
    # @testset "NLTCS SPN" begin
    #     using DataFrames, CSV
    #     # load larger SPN
    #     SPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/nltcs.spn"); offset=1)
    #     @show summary(SPN)
    #     # Load datasets
    #     tdata = convert(Matrix,
    #              DataFrame(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.train.csv"), 
    #                 header=collect(1:16) # columns names
    #     ))) .+ 1;
    #     @show summary(tdata)
    #     vdata = convert(Matrix,
    #                 DataFrame(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.valid.csv"), 
    #                   header=collect(1:16) # columns names
    #     ))) .+ 1;
    #     @show summary(vdata)
    #     # initialize EM learner
    #     learner = EMParamLearner()
    #     initialize(SPN)
    #     # Running Expectation Maximization
    #     while !converged(learner) && learner.steps < 10
    #         if learner.steps % 2 == 0          
    #             tnll = NLL(SPN, vdata)
    #             update(learner, SPN, tdata)
    #             println("It: $(learner.steps) \t NLL: $(learner.score) \t held-out NLL: $tnll")
    #         else
    #             update(learner, SPN, tdata)
    #             println("It: $(learner.steps) \t NLL: $(learner.score)")
    #         end
    #     end

    # end    
end # END of test set

