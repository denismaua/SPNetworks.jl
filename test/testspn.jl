# Test creation and evaluation of SPNs
import SumProductNetworks: ncircuits

@testset "Defining and evaluating discrete SPNs" begin
    @testset "Simple DAG SPN" begin
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
        @test length(SPN) == 8
        @test ncircuits(SPN) == 3
        @test nparams(SPN) == 11
        sc = scope(SPN)
        # println("Scope: ", map(string,sc))
        @test (length(sc) == 2) && (1 in sc) && (2 in sc)
        results = [0.3, 0.15, 0.4, 0.15]
        @testset "Evaluation at $a,$b" for a=1:2, b=1:2
            # v = SPN(a,b)
            @test SPN(a,b) ≈ results[2*(a-1) + b]
            @test logpdf(SPN,[a,b]) ≈ log(SPN([a,b]))
        end
        @testset "Marginalization" begin
            @test SPN(1,NaN) ≈ 0.45
            @test SPN(NaN,2) ≈ 0.3
            @test logpdf(SPN,[NaN,2]) ≈ log(0.3)  
        end
        @testset "Sampling" begin
            x = rand(SPN)
            @test length(x) == 2
            #N = 10000
            N = 1000
            data = rand(SPN,N)
            counts = [0 0; 0 0]
            for i = 1:N
                counts[Int(data[i,1]), Int(data[i,2])] += 1
            end
            @testset "Verifing empirical estimates at $a,$b" for a in 1:2, b in 1:2
                ref = SPN([a,b])
                # println("S($a,$b) = $ref ≈ ", counts[a,b]/N)
                @test ref ≈ counts[a,b]/N atol=0.1
            end                
        end  
    end # end of DAG SPN testset
    @testset "DAG SPN encoding HMM" begin            
        HMM = SumProductNetwork(
            [
            SumNode([2,3],[0.3,0.7]),               # 1
            ProductNode([4,5]),                     # 2
            ProductNode([6,7]),                     # 3
            CategoricalDistribution(1,[0.3,0.7]),   # 4 (D1)
            SumNode([8,9],[0.5,0.5]),               # 5
            SumNode([8,9],[0.2,0.8]),               # 6
            CategoricalDistribution(1,[0.8,0.2]),   # 7 (D2)
            ProductNode([10,11]),                   # 8
            ProductNode([12,13]),                   # 9
            CategoricalDistribution(2,[0.4,0.6]),   # 10 (D3)
            SumNode([14,15],[0.6,0.4]),             # 11
            SumNode([14,15],[0.4,0.6]),             # 12
            CategoricalDistribution(2,[0.25,0.75]), # 13 (D4)
            CategoricalDistribution(3,[0.9,0.1]),   # 14 (D5)
            CategoricalDistribution(3,[0.42,0.58])  # 15 (D6)
            ]
        )
        @test length(HMM) == 15
        @test ncircuits(HMM) == 8
        @test length(scope(HMM)) == 3
        results = [0.11989139999999997,0.06615860000000003,0.29298060000000004,
        0.1709694,0.0708666,0.03658340000000001,0.1561014,0.08644860000000001]
        @testset "Evaluating HMM at $a,$b,$c" for a=1:2, b=1:2, c=1:2
            v = HMM([a,b,c])
            @test v ≈ results[4*(a-1) + 2*(b-1) + c]
        end 
        # println("HMM() ≈ $(HMM([NaN,NaN,NaN]))")
        @test logpdf(HMM,[NaN,NaN,NaN]) ≈ 0.0
        # println("HMM(X1=1) ≈ $(HMM([1,NaN,NaN]))")
        @test logpdf(HMM,[1,NaN,NaN]) ≈ log(0.65)
        x = rand(HMM)
        @test length(x) == 3    
    end # end of HMM testset
    @testset "Selective SPN" begin    
        selSPN = SumProductNetwork(
            [
            SumNode([2,3],[0.4,0.6]),              # 1
            ProductNode([4,5,6]),                  # 2
            ProductNode([7,8,9]),                  # 3
            IndicatorFunction(1,2.0),              # 4
            CategoricalDistribution(2,[0.3,0.7]),  # 5
            CategoricalDistribution(3,[0.4,0.6]),  # 6
            CategoricalDistribution(2,[0.8,0.2]),  # 7
            CategoricalDistribution(3,[0.9,0.1]),  # 8
            IndicatorFunction(1,1.0)               # 9
            ]
        )
        @test length(selSPN) == 9
        @test ncircuits(selSPN) == 2
        @test length(scope(selSPN)) == 3
        results = [0.432,0.048,0.108,0.012,0.048,0.072,0.112,0.168]
        @testset "Evaluating Sel SPN at $a,$b,$c" for a=1:2, b=1:2, c=1:2
            v = selSPN(Float64[a,b,c])
            # println("SEL($a,$b,$c) = $v")
            @test v ≈ results[4*(a-1) + 2*(b-1) + c]
        end    
        # println("- Selective SPN")
        value = selSPN([1,NaN,NaN])
        # println("S2(A=1) = $value")
        @test value ≈ 0.6
        @test logpdf(selSPN,[2,NaN,NaN]) ≈ log(0.4)
        # println()
        x = rand(selSPN)
        @test length(x) == 3    
    end # end of selective SPN testset
    @testset "SPN encoding PSDD" begin            
        # taken from https://github.com/UCLA-StarAI/Circuit-Model-Zoo/blob/master/psdds/little_4var.psdd
        PSDD = SumProductNetwork(
            [
                ProductNode([2,3]),                    # 1 (10)
                SumNode([7,6,5,4],exp.([-1.6094379124341003,-1.2039728043259361,-0.916290731874155,-2.3025850929940455])),     # 2 (9)
                SumNode([11,10,9,8],exp.([-2.3025850929940455,-2.3025850929940455,-2.3025850929940455,-0.35667494393873245])), # 3 (8)
                ProductNode([14,12]), # 4
                ProductNode([14,13]), # 5
                ProductNode([15,12]), # 6
                ProductNode([15,13]), # 7
                ProductNode([18,16]), # 8
                ProductNode([18,17]), # 9
                ProductNode([19,16]), # 10
                ProductNode([19,17]), # 11
                IndicatorFunction(4, 1.0), # 12
                IndicatorFunction(4, 2.0), # 13
                IndicatorFunction(3, 1.0), # 14
                IndicatorFunction(3, 2.0), # 15
                IndicatorFunction(2, 1.0), # 16
                IndicatorFunction(2, 2.0), # 17
                IndicatorFunction(1, 1.0), # 18
                IndicatorFunction(1, 2.0)  # 19
            ]
        )
        @test length(PSDD) == 19
        @test ncircuits(PSDD) == 16
        @test length(scope(PSDD)) == 4
        results = [0.07, 0.27999999999999997, 0.20999999999999996, 0.14, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001, 0.010000000000000005, 0.04000000000000001, 0.029999999999999995, 0.02000000000000001]
        @testset "Evaluating PSDD at $a,$b,$c,$d" for a=1:2, b=1:2, c=1:2, d=1:2
            v = PSDD([a,b,c,d])
            # println("PSDD($a,$b,$c,$d) = $v")
            @test v ≈ results[8*(a-1) + 4*(b-1) + 2*(c-1) + d]
        end    
        @testset "Sampling" begin
            x = rand(PSDD)
            @test length(x) == 4
        end
    end # end of PSDD test
end # END of discrete testset


