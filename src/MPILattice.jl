module MPILattice
using MPI
using LinearAlgebra
using JACC


abstract type Lattice{D,T,AT} end

include("Lattice.jl")
include("Latticematrix.jl")
include("LinearAlgebras/linearalgebra.jl")
include("SpecialUnitary/SU.jl")


#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,shift}
    data::D
end

function Shifted_Lattice(data::Lattice{D,T,AT}, shift) where {D,T,AT}
    return Shifted_Lattice{Lattice{D,T,AT},Tuple(shift)}(data)
end

export Shifted_Lattice



end
