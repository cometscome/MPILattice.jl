module MPILattice
using MPI
using LinearAlgebra
using JACC


include("Lattice.jl")
include("LatticeMatrix.jl")
#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")


end
