struct MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs} <: MLattice1D{T_array,NX,PE,Nwing}
    data::T_array
    PN::NTuple{1,Int64} #number of sites in each process
    myrank::Int64
    wings_back::T_array
    #wings_back_window::MPI.Win
    wings_forward::T_array
    #wings_forward_window::MPI.Win
    comm::MPI.Comm

    function MLattice1Dvector(NC::Integer, NX::Integer, PE::Integer;
        elementtype=Float64,
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD)

        #GC.gc()

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
        #wings_back_window = MPI.Win_create(wings_back, comm)

        wings_forward = zeros(elementtype, NC, Nwing)
        #wings_forward_window = MPI.Win_create(wings_forward, comm)

        T_array = typeof(data)



        return new{T_array,NC,NX,PE,Nwing,Nprocs}(data, PN, myrank,
            wings_back,
            #wings_back_window,
            wings_forward,
            #wings_forward_window,
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

function get_datatype(::MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
    return T_array
end

function Base.getindex(A::MLattice1Dvector{T_array,NC,NX,PE,Nwing,Nprocs}, ic,i::Int) where {T_array,NC,NX,PE,Nwing,Nprocs}
    iout,isinside,isback,isforward = get_localindex(i,A.myrank,A.PN[1],Nwing)
    #println((i,iout,isinside))
    if isinside
        if isback
            v = A.wings_back[ic,iout]
        elseif isforward
            v = A.wings_forward[ic,iout]
        else
            v = A.data[ic,iout]
        end
    else
        v = NaN
    end
    
    return v
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
    #println("myrank = $(A.myrank) $(A.wings_back)")
    MPI.Barrier(A.comm)
    wings_back_window = MPI.Win_create(A.wings_back, A.comm)
    #@time MPI.Win_create(A.wings_back, A.comm)

    MPI.Win_fence(0, wings_back_window)
    #display(A.data)
    #display(wings_back_window)
    #println(view(A.data, 1:NC, (A.PN[1]-Nwing+1):A.PN[1]))
    #println("myrank_sendto $myrank_sendto $(A.myrank)")
    MPI.Put(A.data[ 1:NC, (A.PN[1]-Nwing+1):A.PN[1]], myrank_sendto, wings_back_window)
    MPI.Win_fence(0, wings_back_window)
    MPI.free(wings_back_window)

    #println("myrank = $(A.myrank) $(A.wings_back)")
    MPI.Barrier(A.comm)
    
    #forward wing
    if A.myrank == 0
        myrank_sendto = Nprocs-1
    else
        myrank_sendto = A.myrank - 1
    end

    wings_forward_window = MPI.Win_create(A.wings_forward, A.comm)

    MPI.Win_fence(0,  wings_forward_window)
    MPI.Put(A.data[ 1:NC, 1:Nwing], myrank_sendto,  wings_forward_window)
    MPI.Win_fence(0,  wings_forward_window)

    MPI.free(wings_forward_window)

  # display(A.wings_forward_window)
    #println(A.wings_back[:,:])
    #println(A.wings_forward[:,:])

    MPI.Barrier(A.comm)

    #right wing
end





