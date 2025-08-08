using LinearAlgebra
function LinearAlgebra.mul!(C::Lattice, A::Lattice, B::Lattice)
    error("Matrix multiplication is not implemented for Lattice types $(typeof(A)) $(typeof(B)) $(typeof(C)).")
end

include("linearalgebra_1D.jl")
include("linearalgebra_4D.jl")