# Test creation and evaluation of SPNs

@testset "Loading SPNs from file" begin

    @testset "Simple Categorical SPN" begin
        io = IOBuffer("""#
1 + 2 0.2 3 0.5 4 0.3
2 * 5 7
3 * 5 8
4 * 6 8
# Leaves
5 categorical 1 0.6 0.4
6 categorical 1 0.1 0.9
7 categorical 2 0.3 0.7
8 categorical 2 0.8 0.2
#""")
        S = SumProductNetwork(io)

        @testset "Saving to file then reading it" begin
            SPN = SumProductNetwork([
                SumNode([2,3,4],[0.2,0.5,0.3]),        # 1
                ProductNode([5,7]),                    # 2
                ProductNode([5,8]),                    # 3
                ProductNode([6,8]),                    # 4
                CategoricalDistribution(1, [0.6,0.4]), # 5
                CategoricalDistribution(1, [0.1,0.9]), # 6
                CategoricalDistribution(2, [0.3,0.7]), # 7
                CategoricalDistribution(2, [0.8,0.2])  # 8
            ])
            # @info "## Saving network to disk" 
            SPNetworks.save(SPN,normpath("$(@__DIR__)/../assets/test.spn"))
            # @info "## Loading network from file"
            SPN2 = SumProductNetwork(normpath("$(@__DIR__)/../assets/test.spn"))
            # @info "## Comparing sizes" 
            @test length(SPN) == length(SPN2) == length(S)
            @test nparams(SPN) == nparams(SPN2) == nparams(S)
            @testset "Evaluating simple SPNs at $a,$b" for a in 1:2, b in 1:2
                pred = SPN2([a,b])
                ref = SPN([a,b])
                ref2 = S([a,b])
                @test pred == ref == ref2
            end
        end
    end

    # @info "## Printing graph structure"
    # io = IOBuffer()
    # todot(io,SPN)
    # open("$(@__DIR__)/assets/spn.dot", "w") do f
    #     write(f, String(take!(io)))
    # end
    # println("Call dot -Tpdf assets/spn.dot -o assets/spn.pdf to generate image")

    @testset "XOR SPN" begin
        XOR = SumProductNetwork(normpath("$(@__DIR__)/../assets/xor.spn"))
        @testset "Evaluation XOR at $a,$b,$c" for a=1:2, b=1:2, c=1:2
            v = XOR([a,b,c])
            @test ((a + b + c - 3) % 2 == 1 && v ≈ 0.0) || ((a + b + c - 3) % 2 == 0 && v ≈ 0.25)
        end 
    @test logpdf(XOR,fill(NaN,3)) ≈ 0.0
    end
    
    @testset "HMM SPN"  begin
        HMM = SumProductNetwork(normpath("$(@__DIR__)/../assets/hmm.spn"))
        results = [0.11989139999999997,0.06615860000000003,0.29298060000000004,
                0.1709694,0.0708666,0.03658340000000001,0.1561014,0.08644860000000001]
        @testset "Evaluating HMM at $a,$b,$c" for a=1:2, b=1:2, c=1:2
            v = HMM([a,b,c])
            @test v ≈ results[4*(a-1) + 2*(b-1) + c]
        end 
        @test logpdf(HMM,fill(NaN,3)) ≈ 0.0
    end
    
    @testset "Reading Small SPN in pyspn's format" begin
        PySPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/example.pyspn.spn"); offset = 1)
        @test length(PySPN) == 13
        @test length(scope(PySPN)) == 2
        @test nparams(PySPN) == 12
        @test logpdf(PySPN,[NaN,NaN]) ≈ 0.0 atol=1e-6
        @test PySPN(2,1) ≈ 0.4
        @test logpdf(PySPN, [2,NaN]) ≈ log(0.55)
    end

    @testset "Reading Breast-Cancer SPN in pyspn's format" begin
        PySPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/breast-cancer.spn"); offset = 1)
        @test length(scope(PySPN)) == 10
        @test logpdf(PySPN,fill(NaN,10)) ≈ 0.0 atol=1e-6
    end
end
