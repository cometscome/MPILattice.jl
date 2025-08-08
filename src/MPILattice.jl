module MPILattice
using MPI
using LinearAlgebra
using JACC

export MLattice1Dvector
export MLattice1Dmatrix
export substitute!
export shift_lattice

# Write your package code here.
abstract type MLattice{T_array,Dim} end

abstract type MLattice1D{T_array,NX,PE,Nwing} <: MLattice{T_array,1} end

abstract type MLattice2D{T_array,NX,NY,PEs,Nwing} <: MLattice{T_array,2} end


function parallel_for!(A::MLattice, f::Function, variables...)
    error("parallel_for!: Type $(typeof(A)) is not supported")
end


struct Shifted_1DLattice{T,T_array,shift} <: MLattice{T_array,1}
    data::T
end



include("Lattice.jl")
include("LatticeMatrix.jl")
#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")


end
