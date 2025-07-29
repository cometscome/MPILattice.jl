using MPI
MPI.Init()
using MPI.Types

# ===================== structure ============================================
struct HaloComm1D{T}
    cart      :: MPI.Comm
    sendT     :: NTuple{2,MPI.Datatype}
    recvT     :: NTuple{2,MPI.Datatype}
    sendbuf   :: NTuple{2,Vector{T}}
    counts    :: NTuple{2,Int}
    displs    :: NTuple{2,Int}
    phase     :: T
    Nw        :: Int
end
export HaloComm1D

# ===================== constructor ==========================================
function HaloComm1D(::Type{T}, NC, Nx, Nw; phase::T = one(T)) where {T}
    comm  = MPI.COMM_WORLD
    cart  = MPI.Cart_create(comm, (MPI.Comm_size(comm),), periodic=(true,))
    coords = MPI.Cart_coords(cart, MPI.Comm_rank(cart))
    dims   = MPI.Cartdim_get(cart)

    sizes = (NC, Nx + 2Nw)                          # (c, x)
    sendT = MPI.Datatype[]; recvT = MPI.Datatype[]; sendB = Vector{Vector{T}}()

    for (k, dx) in enumerate((-1, +1))              # west / east
        sub  = (NC, Nw)
        src  = (0, dx == -1 ? Nw         : Nx)      # 0-based!
        dst  = (0, dx == -1 ? 0          : Nx+Nw)   # 0-based!
        push!(sendT, MPI.Types.create_subarray(sizes, sub, src, MPI.DOUBLE) |> MPI.Types.commit!)
        push!(recvT, MPI.Types.create_subarray(sizes, sub, dst, MPI.DOUBLE) |> MPI.Types.commit!)

        boundary = (dx==-1 && coords[1]==0) || (dx==1 && coords[1]==dims[1]-1)
        push!(sendB, boundary ? Vector{T}(undef, prod(sub)) : Vector{T}())
    end
    return HaloComm1D{T}(cart, Tuple(sendT),Tuple(recvT), Tuple(sendB), (1,1), (0,0), phase, Nw)
end

"""
    neighbor_alltoallw!(sendbufs, sendtypes, recvbufs, recvtypes, comm)

Light-weight wrapper around `MPI_Neighbor_alltoallw`.
* `sendbufs`  – Tuple (length = #neigh) of Julia arrays
* `sendtypes` – Tuple of MPI.Datatype (one per neighbour)
* `recvbufs`  – Tuple of destination arrays (typically 1 halo array)
* `recvtypes` – Tuple of MPI.Datatype (one per neighbour)
Counts are all assumed to be 1; displacements are absolute byte addresses
by using `MPI.BOTTOM` as the base pointer (standard trick for W-functions).
"""
function neighbor_alltoallw!(sbufs::NTuple{N,Any},
                             sendtypes::NTuple{N,MPI.Datatype},
                             rbufs ::NTuple{N,Any},
                             recvtypes::NTuple{N,MPI.Datatype},
                             comm ::MPI.Comm) where {N}

    # C arrays ---------------------------------------------------------------
    scounts = fill(MPI.Count(1), N)
    rcounts = scounts

    sdispls = MPI.API.MPI_Aint[ MPI.API.MPI_Aint(pointer(sbufs[i])) for i = 1:N ]
    rdispls = MPI.API.MPI_Aint[ MPI.API.MPI_Aint(pointer(rbufs[i])) for i = 1:N ]

    # Ptr to first element (C expects raw *) ---------------------------------
    ptr_sendcounts = pointer(scounts)
    ptr_senddispls = pointer(sdispls)
    ptr_sendtypes  = pointer(sendtypes)
    ptr_recvcounts = pointer(rcounts)
    ptr_recvdispls = pointer(rdispls)
    ptr_recvtypes  = pointer(recvtypes)

    # Call C API with MPI.BOTTOM as the base address -------------------------
    MPI.API.MPI_Neighbor_alltoallw(MPI.BOTTOM,
                                   ptr_sendcounts, ptr_senddispls, ptr_sendtypes,
                                   MPI.BOTTOM,
                                   ptr_recvcounts, ptr_recvdispls, ptr_recvtypes,
                                   comm)
    return
end
# ===================== one-step exchange ====================================
function exchange!(hc::HaloComm1D, ϕ) 
    # Pack & multiply only boundary faces
    if !isempty(hc.sendbuf[1])                # −x face
        @views hc.sendbuf[1] .= ϕ[:, hc.Nw+1 : 2*hc.Nw]
        hc.sendbuf[1] .*= hc.phase
    end
    if !isempty(hc.sendbuf[2])                # +x face
        @views hc.sendbuf[2] .= ϕ[:, end-2*hc.Nw+1 : end-hc.Nw]
        hc.sendbuf[2] .*= hc.phase
    end

neighbor_alltoallw!( (hc.sendbuf[1], hc.sendbuf[2]),
                     (hc.sendT[1],   hc.sendT[2]),
                     (ϕ, ϕ),                       # recv = same halo array
                     (hc.recvT[1],  hc.recvT[2]),
                     hc.cart)
end