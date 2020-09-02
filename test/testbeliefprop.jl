# Test maxproduct algorithm

@testset "BeliefPropagation" begin
    import SPNetworks.MAP: beliefpropagation!

    @testset "Discrete SPNs" begin

        @testset "DAG SPN where MaxProduct is suboptimal" begin
            
            SPN = SumProductNetwork(
                [
                    SumNode([2,3,4],[0.2,0.5,0.3]),        # 1
                    ProductNode([5,7]),                    # 2
                    ProductNode([5,8]),                    # 3
                    ProductNode([6,8]),                    # 4
                    CategoricalDistribution(1, [0.6,0.4]), # 5
                    CategoricalDistribution(1, [0.1,0.9]), # 6
                    CategoricalDistribution(2, [0.3,0.7]), # 7
                    CategoricalDistribution(2, [0.8,0.2])  # 8
                ]
            )
            evidence = [0.0,0.0]
            bp = beliefpropagation!(evidence,SPN,Set([1,2]); maxiterations=3, verbose = false)
            @test logpdf(SPN,evidence) ≈ log(0.4)
            evidence = [0.0,NaN]
            bp = beliefpropagation!(evidence,SPN,Set([1]); maxiterations=3, verbose = false)
            @test bp ≈ 0.55
        end
     
        @testset "Selective SPN" begin
            SPN = SumProductNetwork(
                [
                    SumNode([2,3],[0.4,0.6]),              # 1
                    ProductNode([4,5,6]),                  # 2
                    ProductNode([7,8,9]),                  # 3
                    IndicatorFunction(1,2.0),  # 4
                    CategoricalDistribution(2,[0.3,0.7]),  # 5
                    CategoricalDistribution(3,[0.4,0.6]),  # 6
                    CategoricalDistribution(2,[0.8,0.2]),  # 7
                    CategoricalDistribution(3,[0.9,0.1]),  # 8
                    IndicatorFunction(1,1.0) # 9
                ]
            )
            lmaxv = -Inf
            for a=1:2,b=1:2,c=1:2
                v = logpdf(SPN,[a,b,c])
                lmaxv = v > lmaxv ? v : lmaxv
            end
            evidence = [0.0,0.0,0.0]
            mp = beliefpropagation!(evidence,SPN,Set([1,2,3]); maxiterations=3, verbose = false)
            @test logpdf(SPN,evidence) ≈ lmaxv
        end
    end
end # END of test set
