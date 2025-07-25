struct MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs} <: MLattice1D{T_array,NX,PE,Nwing}
    data::T_array
    PN::NTuple{1,Int64} #number of sites in each process
    myrank::Int64
    wings_back::T_array
    #wings_back_window::MPI.Win
    wings_forward::T_array
    #wings_forward_window::MPI.Win
    comm::MPI.Comm

    function MLattice1Dmatrix(NC1::Integer,NC2::Integer, NX::Integer, PE::Integer;
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

        data = zeros(elementtype, NC1,NC2, PN[1]+2Nwing)

        wings_back = zeros(elementtype, NC1,NC2, Nwing)
        #wings_back_window = MPI.Win_create(wings_back, comm)

        wings_forward = zeros(elementtype, NC1,NC2, Nwing)
        #wings_forward_window = MPI.Win_create(wings_forward, comm)

        T_array = typeof(data)



        return new{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}(data, PN, myrank,
            wings_back,
            #wings_back_window,
            wings_forward,
            #wings_forward_window,
            comm)
    end

    function MLattice1Dmatrix(A::AbstractArray{T,3}, PE::Integer;
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD) where T

        NC1, NC2, NX = size(A)
        elementtype = eltype(A)

        #GC.gc()
       

        M = MLattice1Dmatrix(NC1, NC2, NX, PE;
            elementtype,
            mpiinit,
            Nwing,
            comm)


        #GC.gc()
        MPI.Barrier(comm)

        for i = 1:M.PN[1]
            ix = get_ix(i, M.myrank, M.PN[1])
            for jc = 1:NC2
                for ic = 1:NC1
                    #println(A[ic, jc,ix],(ic, i+Nwing))
                    M.data[ic, jc,i+Nwing] = A[ic, jc,ix]
                end
            end
        end

        set_wing!(M)

        return M
    end
end

function jaccMPI_kernel_mul!(i, C::AbstractArray{T, 3}, A, B, NC,Nwing) where {T}
    for jc = 1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc=1:NC
                C[ic,jc,i+Nwing] += A[ic,kc,i+Nwing]*B[kc,jc,i+Nwing]
            end
        end
    end
end

function LinearAlgebra.mul!(C::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs},A::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs},
    B::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}) where 
    {T_array,NC1,NC2,NX,PE,Nwing,Nprocs}
    @assert NC1 == NC2 "matrix should be square"
    JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul!,C.data,A.data,B.data,NC1,Nwing)
    set_wing!(C) 
end



function get_datatype(::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}) where {T_array,NC1,NC2,NX,PE,Nwing,Nprocs}
    return T_array
end

function Base.getindex(A::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}, ic,jc,i::Int) where {T_array,NC1,NC2,NX,PE,Nwing,Nprocs}
    iout = get_localindex(i,A.myrank,A.PN[1],Nwing)
    isinside =(iout in 1:(A.PN[1]+2Nwing))
    if isinside
        return A.data[ic,jc,iout]
    else
        a = similar(A.data[1:NC1,1:NC2,1])
        a .= NaN
        return  a
    end

end


function Base.display(A::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}) where {T_array,NC1,NC2,NX,PE,Nwing,Nprocs}
    for myrank_i = 0:Nprocs-1
        #println(myrank_i)
        #MPI.Barrier(A.comm)

        if myrank_i == A.myrank
            println("Process: $(A.myrank)")
            for i = 1:A.PN[1]
                ix = get_ix(i, A.myrank, A.PN[1])
                for jc = 1:NC2
                    for ic = 1:NC1
                        println("$ic \t $ix \t $(A.data[ic,jc,i+Nwing])")
                    end
               end
            end
        end
        MPI.Barrier(A.comm)
    end

    MPI.Barrier(A.comm)
    
end



function set_wing!(A::MLattice1Dmatrix{T_array,NC1,NC2,NX,PE,Nwing,Nprocs}) where {T_array,NC1,NC2,NX,PE,Nwing,Nprocs}
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
    istart = Nwing + A.PN[1] - Nwing + 1
    iend = Nwing +A.PN[1]
    MPI.Put(A.data[ 1:NC1,1:NC2, istart:iend], myrank_sendto, wings_back_window)
    MPI.Win_fence(0, wings_back_window)
    MPI.free(wings_back_window)

    #println("myrank = $(A.myrank) $(A.wings_back)")
    for i=1:Nwing
        for jc=1:NC2
            for ic=1:NC1
                A.data[ic,jc,i] = A.wings_back[ic,jc,i]
            end
        end
    end
    MPI.Barrier(A.comm)
    
    #forward wing
    if A.myrank == 0
        myrank_sendto = Nprocs-1
    else
        myrank_sendto = A.myrank - 1
    end

    wings_forward_window = MPI.Win_create(A.wings_forward, A.comm)
    istart = Nwing + 1
    iend = Nwing + Nwing

    MPI.Win_fence(0,  wings_forward_window)
    MPI.Put(A.data[1:NC1,1:NC2, istart:iend], myrank_sendto,  wings_forward_window)
    MPI.Win_fence(0,  wings_forward_window)

    MPI.free(wings_forward_window)

  # display(A.wings_forward_window)
    #println(A.wings_back[:,:])
    #println(A.wings_forward[:,:])
    for i=1:Nwing
        for jc=1:NC2
            for ic=1:NC1
                A.data[ic,jc,Nwing+A.PN[1]+i] = A.wings_forward[ic,jc,i]
            end
        end
    end

    MPI.Barrier(A.comm)

    #right wing
end





