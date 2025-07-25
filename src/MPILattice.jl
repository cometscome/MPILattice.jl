module MPILattice
using MPI

export MLattice1Dvector

# Write your package code here.
abstract type MLattice{T_array,Dim} end

abstract type MLattice1D{T_array,NX,PE,Nwing} <: MLattice{T_array,1} end

include("1D/1Dlattice.jl")

end
