using MPILattice
using Test
using MPI
import JACC
using LinearAlgebra
using InteractiveUtils
JACC.@init_backend
    using MPI, JACC, StaticArrays

function mpitest()
    #using MPI
    MPI.Init()

    comm  = MPI.COMM_WORLD
    rank  = MPI.Comm_rank(comm)

    # ─ 派生データ型を作成（ここでは配列末尾 3 要素だけ送る例） ─
    A   = rand(Float64, 10)
    subs = [3]                      # 送りたい長さ
    offs = [7]                      # 0-based start index (=7 → 8–10 番目)
    dt   = MPI.Type_create_subarray(1, size(A), subs, offs,
                                    MPI.ORDER_FORTRAN, MPI.Datatype(Float64))
    MPI.Type_commit(dt)

    if rank == 0
        req = MPI.Isend(A, 1, 0, comm; datatype = dt)   # ← 明示指定
        MPI.Wait(req)
    elseif rank == 1
        buf = zeros(Float64, 3)
        req = MPI.Irecv!(buf, 0, 0, comm; datatype = dt)
        MPI.Wait(req)
        @show buf   # ⇒ 送られてきた 8–10 番目の値だけが入る
    end
    MPI.Finalize()
end
#mpitest()
#return

function latticetest()
    MPI.Init()
    NC = 3
    dim = 1
    NX = 8
    NY = 8
    gsize = (NX,)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    PEs = (nprocs,)
    A = Lattice(NC,dim,gsize,PEs)

    set_halo!(A)
end




function halotest()
    MPI.Init()
    NC = 3
    NX = 12
    Nw = 1
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)
    h = HaloComm1D(Float64, NC, NX, Nw)

    PE = Nprocs
    Nwing = 1

    M1 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing,elementtype=ComplexF64)

    display(M1)

    A = zeros(ComplexF64,1, NX)
    A[:] = collect(1:NX)
    println(A)
    B = zeros(ComplexF64,1, NX)
    B[:] = 10 .* collect(1:NX)


    

    M2 = MLattice1Dvector(A, PE,Nwing=Nwing)
    M3 = MLattice1Dvector(B, PE,Nwing=Nwing)
    M4 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing,elementtype=ComplexF64)
    
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

end

function test()
    MPI.Init()
    NC = 1
    NX = 12
    comm = MPI.MPI.COMM_WORLD
    Nprocs = MPI.Comm_size(comm)

    PE = Nprocs
    Nwing = 1

    M1 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing,elementtype=ComplexF64)

    display(M1)

    A = zeros(ComplexF64,1, NX)
    A[:] = collect(1:NX)
    println(A)
    B = zeros(ComplexF64,1, NX)
    B[:] = 10 .* collect(1:NX)


    

    M2 = MLattice1Dvector(A, PE,Nwing=Nwing)
    M3 = MLattice1Dvector(B, PE,Nwing=Nwing)
    M4 = MLattice1Dvector(NC, NX, PE;Nwing=Nwing,elementtype=ComplexF64)
    
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
    mul!(M1,M2,M6)
    display(M1)

    #@code_warntype mul!(M1,M2,M6)

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
    latticetest()
    #halotest()
    #test()
    #test2()
    #test1d()
end
