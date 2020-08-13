# Test maxproduct algorithm
import SumProductNetworks.MAP: maxproduct!

@testset "MaxProduct" begin

    @testset "Discrete SPNs" begin

        @testset "Simple SPN" begin        
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
            values = Vector{Float64}(undef,length(SPN))
            tree = Vector{UInt}(undef,length(SPN))
            mp = maxproduct!(values,tree,SPN,Set([1,2]),[1,1])
            expected = log.([0.24,0.42,0.48,0.72,0.6,0.9,0.7,0.8])
            for (i,v) in enumerate(values)
                @test v ≈ expected[i]
            end
            @test tree[1] == 3
            evidence = [0.0,0.0]
            mp = maxproduct!(evidence,SPN,Set([1,2]))
            @test logpdf(SPN,evidence) ≈ log(0.3)
            evidence = [0.0,NaN]
            mp = maxproduct!(evidence,SPN,Set([1]))
            @test logpdf(SPN,evidence) ≈ log(0.45)
        end

        @testset "Dichotomized version" begin
            SPN = SumProductNetwork([
                SumNode([3,2],[0.2,0.8]),
                SumNode([4,5],[0.625,0.375]),
                ProductNode([6,8]),
                ProductNode([6,9]),
                ProductNode([7,9]),
                CategoricalDistribution(1,[0.6,0.4]),
                CategoricalDistribution(1,[0.1,0.9]),
                CategoricalDistribution(2,[0.3,0.7]),
                CategoricalDistribution(2,[0.8,0.2])
            ]
            )
            evidence = [0.0,0.0]
            mp = maxproduct!(evidence,SPN,Set([1,2]))
            @test logpdf(SPN,evidence) ≈ log(0.3)
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
            mp = maxproduct!(evidence,SPN,Set([1,2,3]))
            @test logpdf(SPN,evidence) ≈ lmaxv
        end
        @testset "Breast-Cancer SPN" begin
            SPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/breast-cancer.spn"), offset=1)
            query = Set([9,1,5])
            evidence = [0,2,NaN,NaN,0,NaN,2,NaN,0,1]
            maxproduct!(evidence,SPN,query)
            @test SPN(evidence) ≈ 0.01125 atol=1e-5
            # println("MAP: ", join(map( q -> "X$q = $(evidence[q])", collect(query)), ", ") ) 
            0.00194
            query = Set([2,5,3])
            evidence = [6,0,0,NaN,0,2,NaN,NaN,NaN,1]
            maxproduct!(evidence,SPN,query)
            @test SPN(evidence) ≈ 0.00194  atol=1e-5
            # println("MAP: ", join(map( q -> "X$q = $(evidence[q])", collect(query)), ", ") ) 
        end
    end
end # END of test set
