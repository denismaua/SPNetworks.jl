# Test evaluation performance of learned SPNs

@testset "Evaluation of large tree-shaped SPNs" begin

    using DataFrames, CSV

    @info "## Loading network from file"
    SPN, totaltime = SumProductNetworks.load(normpath("$(@__DIR__)/../assets/nltcs.learnspn.notuning.spn"))
    println("Loading took ", totaltime, "s")
    println(SPN)

    @info "## Reading test set"
    @time testset = DataFrame!(CSV.File(normpath("$(@__DIR__)/../assets/nltcs.test.csv"), 
                             header=collect(1:16) # columns names
                             ));
    println("Testset has $(size(testset,1)) instances, $(size(testset,2)) columns.")
    test = convert(Matrix, testset) .+ 1
    @info "## Evaluating on test set"
    println("LL: ", logpdf(SPN,test)) 
    @test logpdf(SPN,test) ≈ -21281.999990461
    println("Benchmarking logpdf")
    @btime logpdf($SPN,$test)
    
    @info "## Loading network from file"
    SPN, totaltime = SumProductNetworks.load(normpath("$(@__DIR__)/../assets/nltcs.learnspn.em.spn"))
    println("Loading took ", totaltime, "s")
    println(SPN)

    @info "## Evaluating on test set"
    println("LL: ", logpdf(SPN,test))
    @test logpdf(SPN,test) ≈ -22284.141587948394
    println("Benchmarking logpdf")
    @btime logpdf($SPN,$test)
end
