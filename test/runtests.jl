using MPILattice
using Test
using MPI

function test()
    MPI.Init()
    NC = 1
    NX = 12
    PE = 1

    M1 = MLattice1Dvector(NC, NX, PE)

    A = zeros(1, NX)
    A[:] = collect(1:NX)

    M2 = MLattice1Dvector(A, PE)

    display(M2)

    return true
end

@testset "MPILattice.jl" begin
    # Write your tests here.
    test()
end
