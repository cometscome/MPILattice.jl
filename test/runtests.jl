using MPILattice
using Test
using MPI
import JACC
using LinearAlgebra
JACC.@init_backend

function test()
    MPI.Init()
    NC = 1
    NX = 12
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)

    PE = Nprocs
    Nwing = 1

    M1 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing)

    display(M1)

    A = zeros(1, NX)
    A[:] = collect(1:NX)
    println(A)
    B = zeros(1, NX)
    B[:] = 10 .* collect(1:NX)


    

    M2 = MLattice1Dvector(A, PE,Nwing=Nwing)
    M3 = MLattice1Dvector(B, PE,Nwing=Nwing)
    M4 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing)
    
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

    mul!(M1,M2,M3)

    display(M2)
    display(M3)
    display(M1)



    if M2.myrank == 0
        for ix=1-Nwing:NX+Nwing
            println("$ix $(M1[1:NC,ix])")
        end
    end

    MPI.Barrier(comm)

    if M2.myrank == 1
        for ix=1-Nwing:NX+Nwing
            println("$ix $(M1[1:NC,ix])")
        end
    end

    MPI.Barrier(comm)

    substitute!(M4,M1)
    display(M4)

    M5 = shift_lattice(M1,1)
    substitute!(M4,M5)
    display(M4)

    mul!(M1,M2,M5)
    display(M1)


    M6 = shift_lattice(M1,4)


    return true
end

function test2()
    MPI.Init()
    NC = 2
    NX = 12
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)

    PE = Nprocs

    Nwing = 1

    M1 = MLattice1Dmatrix(NC, NC,NX, PE;Nwing)

    display(M1)

    A = zeros(NC,NC, NX)
    for i=1:NX
        A[:,:,i] .= i
    end
    println(A)

    B = zeros(NC,NC, NX)
    for i=1:NX
        B[:,:,i] .= 10*i
    end
    println(B)


    M2 = MLattice1Dmatrix(A, PE,Nwing=Nwing)
    M3 = MLattice1Dmatrix(B, PE,Nwing=Nwing)
    

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

    mul!(M1,M2,M3)

    display(M2)
    display(M3)
    display(M1)

    if M2.myrank == 0
        for ix=1-Nwing:NX+Nwing
            println(ix)
            display(M1[1:NC,1:NC,ix])
        end
    end

    MPI.Barrier(comm)

    if M2.myrank == 1
        for ix=1-Nwing:NX+Nwing
            println(ix)
            display(M1[1:NC,1:NC,ix])
        end
    end

    return true
end

@testset "MPILattice.jl" begin
    # Write your tests here.
    test()
    #test2()
end
