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

    M1 = MLattice1Dvector(NC, NX, PE;Nwing=2)

    display(M1)

    A = zeros(1, NX)
    A[:] = collect(1:NX)
    println(A)
    Nwing = 3

    M2 = MLattice1Dvector(A, PE,Nwing=Nwing)
    

    display(M2)

    if M2.myrank == 0
        for ix=1-Nwing:NX+Nwing
            println("$ix $(M2[1:NC,ix])")
        end
    end

    MPI.Barrier(comm)

    if M2.myrank == 1
        for ix=1-Nwing:NX+Nwing
            println("$ix $(M2[1:NC,ix])")
        end
    end

    MPI.Barrier(comm)

    return true
end

function test2()
    MPI.Init()
    NC = 2
    NX = 12
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)

    PE = Nprocs

    M1 = MLattice1Dmatrix(NC, NC,NX, PE;Nwing=2)

    display(M1)

    A = zeros(NC,NC, NX)
    for i=1:NX
        A[:,:,i] .= i
    end
    println(A)
    Nwing = 3

    M2 = MLattice1Dmatrix(A, PE,Nwing=Nwing)
    

    display(M2)

    if M2.myrank == 0
        for ix=1-Nwing:NX+Nwing
            println(ix)
            display(M2[1:NC,1:NC,ix])
        end
    end

    MPI.Barrier(comm)

    if M2.myrank == 1
        for ix=1-Nwing:NX+Nwing
            println(ix)
            display(M2[1:NC,1:NC,ix])
        end
    end

    MPI.Barrier(comm)

    return true
end

@testset "MPILattice.jl" begin
    # Write your tests here.
    #test()
    test2()
end
