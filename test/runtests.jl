using MPILattice
using Test
using MPI

function test()
    MPI.Init()
    NC = 1
    NX = 12
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)

    PE = Nprocs

    M1 = MLattice1Dvector(NC, NX, PE)

    display(M1)

    A = zeros(1, NX)
    A[:] = collect(1:NX)
    println(A)

    M2 = MLattice1Dvector(A, PE)

    display(M2)

    return true
end

@testset "MPILattice.jl" begin
    # Write your tests here.
    test()
end
