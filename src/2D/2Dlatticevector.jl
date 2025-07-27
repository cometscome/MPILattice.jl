#include("kernels_1Dlatticevector.jl")

function get_myrank_xy(myrank, PEs)
    #myrank = ((myrank_y)*PEs[1] + myrank_x
    myrank_x = myrank % PEs[1]
    myrank_y = (i - myrank_x) ÷ PEs[1]
    return myrank_x, myrank_y
end

struct MLattice2Dvector{T_array,NC,NX,NY,PEs,Nwing,Nprocs} <: MLattice2D{T_array,NX,NY,PEs,Nwing}
    data::T_array
    PN::NTuple{2,Int64} #number of sites in each process
    myrank::Int64
    myrank_xy::NTuple{2,Int64}
    #wings_back::T_array
    #wings_back_window::MPI.Win
    #wings_forward::T_array
    #wings_forward_window::MPI.Win
    comm::MPI.Comm
    datashifted::T_array

    function MLattice2Dvector(NC::Integer, NX::Integer, NY::Integer,PEs::NTuple{2,Integer};
        elementtype=Float64,
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD)

        #GC.gc()

        if mpiinit == false
            MPI.Init()
        end
        #MPI.Barrier(comm)

        @assert NX % PEs[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs[1] = $(PEs[1])"
        @assert NY % PEs[2] == 0 "NY % PEs[2] should be 0. Now NX = $NY and PEs[1] = $(PEs[2])"
        PNs = (NX ÷ PEs[1],NY ÷ PEs[2])
        Nprocs = MPI.Comm_size(comm)
        @assert prod(PEs) == Nprocs "num. of MPI process should be prod(PEs). Now Nprocs = $Nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)
        myrank_xy = get_myrank_xy(myrank, PEs)

        data = zeros(elementtype, NC, PN[1]+2Nwing,PN[2]+2Nwing)

        #wings_back = zeros(elementtype, NC, Nwing)
        #wings_back_window = MPI.Win_create(wings_back, comm)

        #wings_forward = zeros(elementtype, NC, Nwing)
        #wings_forward_window = MPI.Win_create(wings_forward, comm)

        T_array = typeof(data)

        datashifted = zeros(elementtype, NC, PN[1],PN[2])

        return new{T_array,NC,NX,NY,PEs,Nwing,Nprocs}(data, PNs, myrank,
            myrank_xy,
            #wings_back,
            #wings_back_window,
            #wings_forward,
            #wings_forward_window,
            comm,
            datashifted)
    end

    function MLattice2Dvector(A::AbstractMatrix{T}, PEs::NTuple{2,Integer};
        mpiinit=true,
        Nwing=1,
        comm=MPI.COMM_WORLD) where T

        NC, NX,NY = size(A)
        elementtype = eltype(A)

        #GC.gc()
       

        M = MLattice2Dvector(NC, NX, NY, PEs;
            elementtype,
            mpiinit,
            Nwing,
            comm)


        #GC.gc()
        MPI.Barrier(comm)

        for i2 = 1:M.PN[1]
            iy = get_ix(i2, M.myrank_xy[2], M.PN[2])
            for i1 = 1:M.PN[1]
                ix = get_ix(i1, M.myrank_xy[1], M.PN[1])
                for ic = 1:NC
                    #println(A[ic, ix],(ic, i+Nwing))
                    M.data[ic, ix+Nwing,iy+Nwing] = A[ic, ix,iy]
                end
            end
        end

        set_wing!(M)

        return M
    end
end



function LinearAlgebra.mul!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},
    B::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs}
    JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul!,C.data,A.data,B.data,NC,Nwing)
    set_wing!(C) 
end

function LinearAlgebra.mul!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},
    B::Shifted_1DLattice{MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},T_array,shift}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs,shift}
    if -Nwing <= shift && shift <= Nwing 
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedB!,C.data,A.data,B.data.data,NC,Nwing,shift)
    else
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedcopiedB!,C.data,A.data,B.data.datashifted,NC,Nwing,shift)
    end
    set_wing!(C) 
end

function LinearAlgebra.mul!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},
    A::Shifted_1DLattice{MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},T_array,shift},
    B::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs,shift}
    if -Nwing <= shift && shift <= Nwing 
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedA!,C.data,A.data.data,B.data,NC,Nwing,shift)
    else
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedcopiedA!,C.data,A.data.datashifted,B.data,NC,Nwing,shift)
    end
    set_wing!(C) 
end

function LinearAlgebra.mul!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},
    A::Shifted_1DLattice{MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},T_array,shift1},
    B::Shifted_1DLattice{MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},T_array,shift2}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs,shift1,shift2}
    if -Nwing <= shift1 && shift1 <= Nwing 
        if -Nwing <= shift2 && shift2 <= Nwing 
            JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedAshiftedB!,C.data,A.data.data,B.data.data,NC,Nwing,shift1,shift2)
        else
            JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedAshiftedcopiedB!,C.data,A.data.data,B.data.datashifted,NC,Nwing,shift1)
        end
    else
        if -Nwing <= shift2 && shift2 <= Nwing 
            JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedcopiedAshiftedB!,C.data,A.data.datashifted,B.data.data,NC,Nwing,shift2)
        else
            JACC.parallel_for(C.PN[1],jaccMPI_kernel_mul_shiftedcopiedAshiftedcopiedB!,C.data,A.data.datashifted,B.data.datashifted,NC,Nwing)
        end
    end
    set_wing!(C) 
end

function substitute!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs}
    JACC.parallel_for(C.PN[1],jaccMPI_kernel_substitute!,C.data,A.data,NC,Nwing)
    set_wing!(C) 
end

function substitute!(C::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},
    A::Shifted_1DLattice{MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},T_array,shift}) where 
    {T_array,NC,NX,PE,Nwing,Nprocs,shift}
    if -Nwing <= shift && shift <= Nwing 
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_substitute_shiftedA!,C.data,A.data.data,NC,Nwing,shift)
    else
        JACC.parallel_for(C.PN[1],jaccMPI_kernel_substitute_shiftedcopiedA!,C.data,A.data.datashifted,NC,Nwing,shift)
    end
    set_wing!(C) 
end

function shift_lattice(data::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},shift) where 
        {T_array,NC,NX,PE,Nwing,Nprocs} 
    if abs(shift) > Nwing
        make_shifteddata!(data,shift) 
    end
    return Shifted_1DLattice{typeof(data),T_array,shift}(data)
end

function check_rank(ix,PN)
    
    #ix = i + PN * myrank
    i = mod(ix-1,PN)+1
    rank = div(ix-i,PN)
    return rank,i
end

function make_shifteddata!(data::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs},shift) where 
        {T_array,NC,NX,PE,Nwing,Nprocs} 
    win = MPI.Win_create(data.datashifted, data.comm)

    MPI.Win_fence(0, win)
   
    for ix=1:data.PN[1]
        ixp = ix - shift
        ip = ixp + data.PN[1] * data.myrank
        while ip > NX
            ip -= NX
        end
        while ip < 1
            ip += NX
        end
        rank,locali = check_rank(ip,data.PN[1])
        MPI.Put(data.data[:, ix+Nwing], rank, (locali-1)*NC,win)
    end

    MPI.Win_fence(0, win)
    MPI.free(win)
end



function get_datatype(::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
    return T_array
end

function Base.getindex(A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}, ic,i::Int) where {T_array,NC,NX,PE,Nwing,Nprocs}
    iout = get_localindex(i,A.myrank,A.PN[1],Nwing)
    isinside =(iout in 1:(A.PN[1]+2Nwing))
    if isinside
        return A.data[ic,iout]
    else
        a = similar(A.data[1:NC,1])
        a .= NaN
        return  a
    end

end


function Base.display(A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
    for myrank_i = 0:Nprocs-1
        #println(myrank_i)
        #MPI.Barrier(A.comm)

        if myrank_i == A.myrank
            println("Process: $(A.myrank)")
            for i = 1:A.PN[1]
                ix = get_ix(i, A.myrank, A.PN[1])
                for ic = 1:NC
                    println("$ic \t $ix \t $(A.data[ic,i+Nwing])")
                end
            end
        end
        MPI.Barrier(A.comm)
    end

    MPI.Barrier(A.comm)
    
end



function set_wing!(A::MLattice2Dvector{T_array,NC,NX,PE,Nwing,Nprocs}) where {T_array,NC,NX,PE,Nwing,Nprocs}
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
    MPI.Put(A.data[ 1:NC, istart:iend], myrank_sendto, wings_back_window)
    MPI.Win_fence(0, wings_back_window)
    MPI.free(wings_back_window)

    #println("myrank = $(A.myrank) $(A.wings_back)")
    for i=1:Nwing
        for ic=1:NC
            A.data[ic,i] = A.wings_back[ic,i]
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
    MPI.Put(A.data[ 1:NC, istart:iend], myrank_sendto,  wings_forward_window)
    MPI.Win_fence(0,  wings_forward_window)

    MPI.free(wings_forward_window)

  # display(A.wings_forward_window)
    #println(A.wings_back[:,:])
    #println(A.wings_forward[:,:])
    for i=1:Nwing
        for ic=1:NC
            A.data[ic,Nwing+A.PN[1]+i] = A.wings_forward[ic,i]
        end
    end

    MPI.Barrier(A.comm)

    #right wing
end





