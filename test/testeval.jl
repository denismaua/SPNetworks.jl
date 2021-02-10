# Test evaluation performance of learned SPNs

@testset "Evaluation of large tree-shaped SPNs" begin

    using DataFrames, CSV

    SPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/nltcs.spn"))

    testset = DataFrame!(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.test.csv"), 
                             header=collect(1:16) # columns names
                             ));
    # println("Testset has $(size(testset,1)) instances, $(size(testset,2)) columns.")
    test = convert(Matrix, testset) .+ 1
    @test logpdf(SPN,test) ≈ -19582.020235794218 #-21281.999990461
    @test plogpdf(SPN,test) ≈ -19582.020235794218 #-21281.999990461
    # @btime logpdf($SPN,$test)
    
    # SPN = SumProductNetwork(normpath("$(@__DIR__)/../assets/nltcs.learnspn.em.spn"))
    # @test logpdf(SPN,test) ≈ -22284.141587948394
    # @btime logpdf($SPN,$test)
end
