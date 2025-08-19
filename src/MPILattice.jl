module MPILattice
using MPI
using LinearAlgebra
using JACC


abstract type Lattice{D,T,AT} end



#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,shift}
    data::D
end

function Shifted_Lattice(data::Lattice{D,T,AT}, shift) where {D,T,AT}
    return Shifted_Lattice{typeof(data),Tuple(shift)}(data)
end

export Shifted_Lattice

struct Adjoint_Lattice{D}
    data::D
end

function Base.adjoint(data::Lattice{D,T,AT}) where {D,T,AT}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::Shifted_Lattice{D,shift}) where {D,shift}
    return Adjoint_Lattice{typeof(data)}(data)
end

include("Lattice.jl")
include("Latticematrix.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")



end
