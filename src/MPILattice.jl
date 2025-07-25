module MPILattice
using MPI

export MLattice1Dvector

# Write your package code here.
abstract type MLattice{T_array,Dim} end

abstract type MLattice1D{T_array,NX,PE,Nwing} <: MLattice{T_array,1} end

struct Shifted_1DLattice{T,T_array,shift} <: MLattice{T_array,1}
    data::T
end


function shift_lattice(data::MLattice1D,shift)
    T_array = get_datatype(data)
    return Shifted_1DLattice{typeof(data),T_array,shift}(data)
end

export shift_lattice

include("1D/1Dlattice.jl")

function get_ix(i, myrank, PN)
    ix = i + PN * myrank
    return ix
end

function check_index(i,NLX,Nwing)
    #println(i)
    isinside = (i in (1-Nwing):(NLX+Nwing))
    isback = (i in 1-Nwing:0)
    
    isforward = (i in NLX+1:NLX+Nwing)
    if isinside
        if isback
            iout = i + Nwing
        elseif isforward
            iout = i - NLX
        else
            iout = i
        end
    else
        iout = 0
    end

    #println((iout,isinside,isback,isforward))
    return iout,isinside,isback,isforward
end

function get_localindex(ix,myrank,PN,Nwing)
    i = ix - PN * myrank
    iout,isinside,isback,isforward = check_index(i,PN,Nwing)
    return iout,isinside,isback,isforward
end

end
