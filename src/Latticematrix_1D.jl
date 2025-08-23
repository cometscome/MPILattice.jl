##############################################################################
#  LatticeMatrix (no derived datatypes, LATTICE = 1D FLATTENED)
#  --------------------------------------
#  * column-major layout   :  (NC1, NC2, S)  where S = prod(local sizes with halos)
#  * halo width            :  nw
#  * per–direction phases  :  φ
#  * internal DoF          :  NC1, NC2  (both faster than lattice dim)
#  * ALWAYS packs faces into contiguous buffers and sends them as plain arrays.
#  * Works for any spatial dimension D, while physical storage is always 3D.
##############################################################################

using MPI, StaticArrays, JACC

struct LatticeMatrix{D,T,AT,NC1,NC2,nw} <: Lattice{D,T,AT}
    nw::Int                          # ghost width
    phases::SVector{D,T}             # per-direction phases
    NC1::Int
    NC2::Int
    gsize::NTuple{D,Int}             # global size (no halos)
    cart::MPI.Comm
    coords::NTuple{D,Int}
    dims::NTuple{D,Int}
    nbr::NTuple{D,NTuple{2,Int}}
    A::AT                            # main array: (NC1, NC2, S)
    buf::Vector{AT}                  # contiguous work buffers (minus/plus, send/recv)
    myrank::Int
    PN::NTuple{D,Int}                # local interior sizes (no halos)
    comm::MPI.Comm
    locS::NTuple{D,Int}              # local sizes WITH halos
    stride::NTuple{D,Int}            # 1D strides over locS
end

# compute strides for flattened spatial index
@generated function _make_stride(locS::NTuple{D,Int}) where {D}
    quote
        NTuple{$D,Int}(i -> (i == 1 ? 1 : prod(locS[1:i-1])))
    end
end

# linear offset (1-based) from D-dim indices (ix₁..ix_D), each in 1:locS[d]
@inline function _lin(ix::NTuple{D,Int}, stride::NTuple{D,Int}) where {D}
    s = 0
    @inbounds @simd for d in 1:D
        s += (ix[d] - 1) * stride[d]
    end
    return s + 1
end

# constructor
function LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64,
    phases=ones(dim), comm0=MPI.COMM_WORLD)
    D = dim
    T = elementtype
    dims = PEs
    cart = MPI.Cart_create(comm0, dims; periodic=ntuple(_ -> true, D))
    coords = MPI.Cart_coords(cart, MPI.Comm_rank(cart))
    nbr = ntuple(d -> ntuple(s -> MPI.Cart_shift(cart, d - 1, s == 1 ? -1 : 1)[2], 2), D)

    # local interior sizes (no halos)
    PN = ntuple(i -> gsize[i] ÷ dims[i], D)
    # local sizes with halos
    locS = ntuple(i -> PN[i] + 2nw, D)
    stride = _make_stride(locS)

    # flattened spatial length
    S = prod(locS)

    # storage is ALWAYS 3D: (NC1, NC2, S)
    A = JACC.zeros(T, NC1, NC2, S)

    # contiguous buffers for each face (send/recv × minus/plus)
    # For a face orthogonal to dim d: slab length = nw * prod(locS)/locS[d]
    function slab_len(d)
        (nw * (S ÷ locS[d]))
    end
    buf = Vector{typeof(A)}(undef, 4D)
    for d in 1:D
        L = slab_len(d)
        buf[4d-3] = JACC.zeros(T, NC1, NC2, L)  # send minus
        buf[4d-2] = JACC.zeros(T, NC1, NC2, L)  # recv minus
        buf[4d-1] = JACC.zeros(T, NC1, NC2, L)  # send plus
        buf[4d] = JACC.zeros(T, NC1, NC2, L)  # recv plus
    end

    return LatticeMatrix{D,T,typeof(A),NC1,NC2,nw}(nw, phases, NC1, NC2, gsize,
        cart, Tuple(coords), dims, nbr, A, buf, MPI.Comm_rank(cart), PN, comm0, locS, stride)
end

# wrap constructor from a global array A (shape = (NC1, NC2, gsize...))
function LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)
    NC1, NC2, NN... = size(A)
    T = eltype(A)
    @assert dim == length(NN)
    gsize = Tuple(NN)

    ls = LatticeMatrix(NC1, NC2, dim, gsize, PEs; elementtype=T, nw, phases, comm0)

    # Copy global interior block to each rank's interior (no halos)
    # We temporarily reshape host-side to (NC1, NC2, locS...) for easy slicing.
    Acpu = Array(ls.A)  # shape (NC1,NC2,S)
    AcpuND = reshape(Acpu, (NC1, NC2, ls.locS...))
    # destination interior indices on local array (with halos)
    dest_idx = ntuple(i -> i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2]), dim + 2)
    # source global ranges for this rank's coordinates
    coords_r = ls.coords
    src_ranges = ntuple(d -> (coords_r[d]*ls.PN[d]+1):(coords_r[d]*ls.PN[d]+ls.PN[d]), D)
    src_idx = (Colon(), Colon(), src_ranges...)

    @views AcpuND[dest_idx...] .= Array(A[src_idx...])  # copy from provided global A
    ls.A .= JACC.array(reshape(AcpuND, size(AcpuND, 1), size(AcpuND, 2), :))

    set_halo!(ls)
    return ls
end

Base.similar(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2} =
    LatticeMatrix(NC1, NC2, D, ls.gsize, ls.dims; nw=ls.nw, elementtype=T, phases=ls.phases, comm0=ls.comm)

# --------------------- packing / unpacking helpers -------------------------

# iterate over all multi-indices except along d; fill ix[d]=rng and copy lines
function _pack_face!(dst::AbstractArray{T,3}, A::AbstractArray{T,3}, nw::Int,
    d::Int, side::Symbol,
    locS::NTuple{D,Int}, stride::NTuple{D,Int}) where {T,D}
    # face range inside interior (width nw)
    rng_d = if side === :minus
        (nw+1):(2*nw)
    else
        (locS[d]-2*nw+1):(locS[d]-nw)
    end
    # number of lines per "hyper-slab"
    lines_per_hyperslab = length(rng_d)

    # total repeats over all other dims
    repeats = prod(locS) ÷ locS[d]

    # fast loop over all sites with ix[d] in rng_d
    # fill dst[:, :, k] incrementally
    k = 0
    @inbounds begin
        # multi-index state (1-based, includes halos)
        idx = ntuple(i -> 1, D)
        # nestless counter over lexicographic order with d as inner-most? no:
        # we hold others as outer loops and sweep ix[d] across rng_d as inner.
        function _advance!(ix)
            for t in 1:D
                if t == d
                    continue
                end
                if ix[t] < locS[t]
                    return Base.setindex(ix, ix[t] + 1, t)
                else
                    ix = Base.setindex(ix, 1, t)
                end
            end
            return nothing
        end
        ix = idx
        rep = 0
        while rep < repeats
            # lock other dims of ix; only vary d across rng_d
            for vd in rng_d
                ix_d = Base.setindex(ix, vd, d)
                base = _lin(ix_d, stride)
                k += 1
                @views dst[:, :, k] .= A[:, :, base]
            end
            rep += 1
            nx = _advance!(ix)
            nx === nothing && break
            ix = nx
        end
    end
    return nothing
end

function _unpack_ghost!(A::AbstractArray{T,3}, src::AbstractArray{T,3}, nw::Int,
    d::Int, side::Symbol,
    locS::NTuple{D,Int}, stride::NTuple{D,Int}) where {T,D}
    rng_d = if side === :minus
        1:nw
    else
        (locS[d]-nw+1):locS[d]
    end
    repeats = prod(locS) ÷ locS[d]
    k = 0
    @inbounds begin
        idx = ntuple(i -> 1, D)
        function _advance!(ix)
            for t in 1:D
                if t == d
                    continue
                end
                if ix[t] < locS[t]
                    return Base.setindex(ix, ix[t] + 1, t)
                else
                    ix = Base.setindex(ix, 1, t)
                end
            end
            return nothing
        end
        ix = idx
        rep = 0
        while rep < repeats
            for vd in rng_d
                ix_d = Base.setindex(ix, vd, d)
                base = _lin(ix_d, stride)
                k += 1
                @views A[:, :, base] .= src[:, :, k]
            end
            rep += 1
            nx = _advance!(ix)
            nx === nothing && break
            ix = nx
        end
    end
    return nothing
end

# pointwise phase multiply on buffer
@inline function _mul_phase!(B::AbstractArray, φ)
    @inbounds @simd for i in eachindex(B)
        B[i] *= φ
    end
end

# NEW: Only multiply if φ is not equal to 1
@inline function _maybe_mul_phase!(B::AbstractArray, φ)
    isone(φ) && return nothing
    _mul_phase!(B, φ)
end


# ----------------------------- halo exchange -------------------------------

function set_halo!(ls::LatticeMatrix{D,T,AT}) where {D,T,AT}
    for id = 1:D
        exchange_dim!(ls, id)
    end
end
export set_halo!

function exchange_dim!(ls::LatticeMatrix{D}, d::Int) where D
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]

    rankM, rankP = ls.nbr[d]
    me = ls.myrank
    reqs = MPI.Request[]

    # NEW: if all phases are one, skip all multiplications
    skip_phase = all(isone, ls.phases)

    φ = ls.phases[d]

    # --- self-neighbor case ---
    if rankM == me && rankP == me
        _pack_face!(bufRM, ls.A, ls.nw, d, :plus, ls.locS, ls.stride)
        _unpack_ghost!(ls.A, bufRM, ls.nw, d, :minus, ls.locS, ls.stride)
        if !skip_phase
            _mul_phase!(_slice_all(bufRM), φ)
        end   # <--- guarded

        _pack_face!(bufRP, ls.A, ls.nw, d, :minus, ls.locS, ls.stride)
        _unpack_ghost!(ls.A, bufRP, ls.nw, d, :plus, ls.locS, ls.stride)
        if !skip_phase
            _mul_phase!(_slice_all(bufRP), φ)
        end   # <--- guarded

        compute_interior!(ls)
        return
    end

    # --- minus direction ---
    if rankM == me
        _pack_face!(bufRM, ls.A, ls.nw, d, :minus, ls.locS, ls.stride)
        if !skip_phase && ls.coords[d] == 0
            _mul_phase!(bufRM, φ)
        end
        _unpack_ghost!(ls.A, bufRM, ls.nw, d, :minus, ls.locS, ls.stride)
    else
        _pack_face!(bufSM, ls.A, ls.nw, d, :minus, ls.locS, ls.stride)
        if !skip_phase && ls.coords[d] == 0
            _mul_phase!(bufSM, φ)
        end
        push!(reqs, MPI.Isend(bufSM, rankM, d, ls.cart))
        push!(reqs, MPI.Irecv!(bufRM, rankM, d + D, ls.cart))
    end

    # --- plus direction ---
    if rankP == me
        _pack_face!(bufRP, ls.A, ls.nw, d, :plus, ls.locS, ls.stride)
        if !skip_phase && ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(bufRP, φ)
        end
        _unpack_ghost!(ls.A, bufRP, ls.nw, d, :plus, ls.locS, ls.stride)
    else
        _pack_face!(bufSP, ls.A, ls.nw, d, :plus, ls.locS, ls.stride)
        if !skip_phase && ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(bufSP, φ)
        end
        push!(reqs, MPI.Isend(bufSP, rankP, d + D, ls.cart))
        push!(reqs, MPI.Irecv!(bufRP, rankP, d, ls.cart))
    end

    compute_interior!(ls)
    isempty(reqs) || MPI.Waitall!(reqs)

    if rankM != me
        _unpack_ghost!(ls.A, bufRM, ls.nw, d, :minus, ls.locS, ls.stride)
    end
    if rankP != me
        _unpack_ghost!(ls.A, bufRP, ls.nw, d, :plus, ls.locS, ls.stride)
    end
end

# tiny helper for broadcasting phase over whole 3D buffer
@inline _slice_all(B) = @view B[:, :, :]

# ---------------------------------------------------------------------------
# hooks (user overrides)
# ---------------------------------------------------------------------------
compute_interior!(::LatticeMatrix) = nothing
compute_boundary!(::LatticeMatrix) = nothing

export LatticeMatrix

# ------------------------------- reductions --------------------------------

function allsum(ls::LatticeMatrix)
    local_sum = sum(ls.A)            # sum EVERYTHING local (including halos)
    MPI.Reduce(local_sum, MPI.SUM, 0, ls.cart)
end
export allsum

# ----------------------------- gather (global) -----------------------------

# Helper: return host Array shaped as (NC1,NC2,locS...) for convenient slicing
@inline function _hostND(ls::LatticeMatrix)
    Acpu = Array(ls.A)                                  # (NC1,NC2,S)
    reshape(Acpu, (ls.NC1, ls.NC2, ls.locS...))         # (NC1,NC2,locS...)
end

function gather_matrix(ls::LatticeMatrix{D,T,AT,NC1,NC2}; root::Int=0) where {D,T,AT,NC1,NC2}
    comm = ls.cart
    me = ls.myrank
    nprocs = MPI.Comm_size(comm)

    # local interior (no halos) on HOST with D-dim view
    AcpuND = _hostND(ls)
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_block_cpu = Array(AcpuND[interior_idx...])        # (NC1,NC2,PN...)

    sendbuf = reshape(local_block_cpu, :)
    if me == root
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        # place root
        function _place_block!(G, blk, coords::NTuple{D,Int})
            ranges = ntuple(d -> (coords[d]*ls.PN[d]+1):(coords[d]*ls.PN[d]+ls.PN[d]), D)
            idx = (Colon(), Colon(), ranges...)
            @views G[idx...] .= blk
        end
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        # receive others
        recvbuf = similar(sendbuf)
        for r in 0:nprocs-1
            r == root && continue
            MPI.Recv!(recvbuf, r, 900, comm)
            coords_r = Tuple(MPI.Cart_coords(comm, r))
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
        return G
    else
        MPI.Send(sendbuf, root, 900, comm)
        return nothing
    end
end
export gather_matrix

function gather_and_bcast_matrix(ls::LatticeMatrix{D,T,AT,NC1,NC2}; root::Int=0) where {D,T,AT,NC1,NC2}
    comm = ls.cart
    me = ls.myrank
    nprocs = MPI.Comm_size(comm)

    AcpuND = _hostND(ls)
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_block_cpu = Array(AcpuND[interior_idx...])
    sendbuf = reshape(local_block_cpu, :)

    G = nothing
    if me == root
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        function _place_block!(G, blk, coords::NTuple{D,Int})
            ranges = ntuple(d -> (coords[d]*ls.PN[d]+1):(coords[d]*ls.PN[d]+ls.PN[d]), D)
            idx = (Colon(), Colon(), ranges...)
            @views G[idx...] .= blk
        end
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        recvbuf = similar(sendbuf)
        for r in 0:nprocs-1
            r == root && continue
            MPI.Recv!(recvbuf, r, 900, comm)
            coords_r = Tuple(MPI.Cart_coords(comm, r))
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
    else
        MPI.Send(sendbuf, root, 900, comm)
    end

    gshape = (ls.NC1, ls.NC2, ls.gsize...)
    if me != root
        G = Array{T}(undef, gshape)
    end
    MPI.Bcast!(G, root, comm)
    return G
end
export gather_and_bcast_matrix