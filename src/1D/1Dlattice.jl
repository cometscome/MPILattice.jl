struct MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs} <: MLattice1D{T_array,NX,PE,Nwing}
    data::T_array
    PN::NTuple{1,Int64} #number of sites in each process
    myrank::Int64
    wings_back::T_array
    wings_back_window::MPI.Win
    wings_forward::T_array
    wings_forward_window::MPI.Win
    comm::MPI.Comm

    function MLattice1Dvector(NC::Integer, NX::Integer, PE::Integer;
        elementtype=Float64,
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD)

        GC.gc()

        if mpiinit == false
            MPI.Init()
        end
        #MPI.Barrier(comm)

        @assert NX % PE == 0 "NX % PE should be 0. Now NX = $NX and PE = $PE"
        PN = (NX ÷ PE,)
        Nprocs = MPI.Comm_size(comm)
        @assert PE == Nprocs "num. of MPI process should be PE. Now Nprocs = $Nprocs and PE = $PE"
        myrank = MPI.Comm_rank(comm)

        data = zeros(elementtype, NC, PN[1])

        wings_back = zeros(elementtype, NC, Nwing)
        wings_back_window = MPI.Win_create(wings_back, comm)

        wings_forward = zeros(elementtype, NC, Nwing)
        wings_forward_window = MPI.Win_create(wings_forward, comm)

        T_array = typeof(data)



        return new{T_array,NC,NX,PE,Nwing,Nprocs}(data, PN, myrank,
            wings_back,
            wings_back_window,
            wings_forward,
            wings_forward_window,
            comm)
    end

    function MLattice1Dvector(A::AbstractMatrix{T}, PE::Integer;
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD) where T

        NC, NX = size(A)
        elementtype = eltype(A)

        #GC.gc()
       

        M = MLattice1Dvector(NC, NX, PE;
            elementtype,
            mpiinit,
            Nwing,
            comm)


        #GC.gc()
        MPI.Barrier(comm)

        for i = 1:M.PN[1]
            ix = get_ix(i, M.myrank, M.PN[1])
            for ic = 1:NC
                M.data[ic, i] = A[ic, ix]
            end
        end

        set_wing!(M)

        return M
    end
end

function get_ix(i, myrank, PN)
    ix = i + PN * myrank
    return ix
end

function Base.display(A::MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
    for myrank_i = 0:Nprocs-1
        #println(myrank_i)
        #MPI.Barrier(A.comm)

        if myrank_i == A.myrank
            println("Process: $(A.myrank)")
            for i = 1:A.PN[1]
                ix = get_ix(i, A.myrank, A.PN[1])
                for ic = 1:NC
                    println("$ic \t $ix \t $(A.data[ic,i])")
                end
            end
        end
        MPI.Barrier(A.comm)
    end

    MPI.Barrier(A.comm)
    
end



function set_wing!(A::MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
    #back wing
    if A.myrank == Nprocs - 1
        myrank_sendto = 0
    else
        myrank_sendto = A.myrank + 1
    end

    #GC.gc()
    MPI.Barrier(A.comm)

    MPI.Win_fence(0, A.wings_back_window)
    MPI.Put(view(A.data, 1:NC, 1:Nwing), myrank_sendto, A.wings_back_window)
    MPI.Win_fence(0, A.wings_back_window)

    
    #forward wing
    if A.myrank == 0
        myrank_sendto = Nprocs-1
    else
        myrank_sendto = A.myrank - 1
    end

    MPI.Win_fence(0,  A.wings_forward_window)
    MPI.Put(view(A.data, 1:NC, 1:Nwing), myrank_sendto,  A.wings_forward_window)
    MPI.Win_fence(0,  A.wings_forward_window)



    MPI.Barrier(A.comm)

    #right wing
end

struct Shifted_1DLattice{T,shift}
    data::T
end
