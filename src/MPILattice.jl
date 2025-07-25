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

end
