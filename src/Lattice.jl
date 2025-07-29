using StaticArrays
# -----------------------------------------------------------------------------  
# Main container
# -----------------------------------------------------------------------------
struct Lattice{D,T,TA,T1}
    nw       ::Int                  # ghost width
    phases        ::SVector{D,T1}         # per-direction phase
    NC       ::Int                  # internal DoF per site
    gsize    ::NTuple{D,Int}        # global lattice size
    cart     ::MPI.Comm             # Cartesian communicator
    coords   ::NTuple{D,Int}        # my coordinates
    dims     ::NTuple{D,Int}        # Px×Py(×Pz…)
    nbr      ::NTuple{D,NTuple{2,Int}} # neighbor ranks [dim][±]
    A        ::TA # main field (device or host)
    buf      ::Vector{TA} # send/recv buffers (± per dim)
    types    ::Dict{Tuple{Int,Symbol},MPI.Datatype} # face datatypes
    myrank   ::Int


    function Lattice(NC,dim,gsize,PEs;nw=1,elementtype=ComplexF64,phases=ones(dim),comm=MPI.COMM_WORLD) 
        D = dim
        dims = PEs
        @assert length(phases) == D
        cart  = MPI.Cart_create(comm, dims; periodic=ntuple(_->true,D))
        coords= MPI.Cart_coords(cart, MPI.Comm_rank(cart))
        nbr   = ntuple(d->ntuple(s->MPI.Cart_shift(cart,d-1,s-1)[2],2),D)
        types = Dict{Tuple{Int,Symbol},MPI.Datatype}()

        locS = ntuple(i->gsize[i] ÷ dims[i] + 2nw, D)
        loc  = (NC,locS...) 
        A = JACC.zeros(elementtype,loc...)
        myrank = MPI.Comm_rank(comm)

        #per-dim send/recv buffers (±)
        # Buffers (minus/plus per dim)
        buf = Vector{typeof(A)}(undef, 2D)
        for d in 1:D
            shp = ntuple(i-> i==d ? nw : locS[i], D)
            buf[2d-1] = JACC.zeros(eltype(phases), (NC, shp...)...) # minus
            buf[2d  ] = JACC.zeros(eltype(phases), (NC, shp...)...) # plus
        end

        #=
        buf = Vector{typeof(A)}(undef, 2D)
        for d in 1:D
            shape = ntuple(i->i==d ? nw : locS[i], D)
            push!(buf,  JACC.zeros(elementtype, shape..., NC)) # minus
            push!(buf,  JACC.zeros(elementtype, shape..., NC)) # plus
        end
        =#

        #MPI datatypes for faces
        for d in 1:D, side in (:minus,:plus)
            sz    = size(A)
            subs  = collect(sz); subs[d] = nw
            offs  = fill(0,D+1)
            offs[d] = (side==:minus ? nw : sz[d]-2nw)
            T= MPI.Types.create_subarray(sz, subs, offs,MPI.Datatype(elementtype))
            MPI.Types.commit!(T)
            types[(d,side)] = T
        end
        T1 = eltype(phases)
        phases_s = SVector{dim}(phases)
        
        

        return new{D,elementtype,typeof(A),T1}(
            nw, 
            phases_s, 
            NC, 
            gsize, 
            cart,
            Tuple(coords),
            dims,
            nbr,
            A,
            buf,
            types,
            myrank
            )
    end
end



@inline apply_phase!(buf, phase) =
    JACC.parallel_for(length(buf)) do i; buf[i] *= phase; end


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
"""
    _face_view(A, nw, d, side)

Return a `@view` of the face (ghost–1 layer) in spatial dimension `d`
(`d = 1 … D`) on the given `side` (`:minus` or `:plus`).

* Array layout is `(NC, X, Y, Z, …)`, so the spatial dim maps to `d+1`.
"""
function _face_view(A, nw, d, side::Symbol)
    idx = ntuple(_ -> Colon(), ndims(A))             # (:, :, :, …)
    if side === :minus
        rng = (nw + 1):(2nw)
    else
        sz  = size(A, d + 1)
        rng = (sz - 2nw + 1):(sz - nw)
    end
    idx = Base.setindex(idx, rng, d + 1)             # replace d+1-th dim
    @views return A[idx...]
end

"""
    _ghost_view(A, nw, d, side)

Return a `@view` of the ghost cells **inside** the domain
(first `nw` or last `nw` layers) for dimension `d`.
"""
function _ghost_view(A, nw, d, side::Symbol)
    idx = ntuple(_ -> Colon(), ndims(A))
    if side === :minus
        rng = 1:nw
    else
        sz  = size(A, d + 1)
        rng = (sz - nw + 1):sz
    end
    idx = Base.setindex(idx, rng, d + 1)
    @views return A[idx...]
end
function set_halo!(ls::Lattice{D,T,TA,T1}) where {D,T,TA,T1}
    for id=1:D
        exchange_dim!(ls, id)
    end
end
export set_halo!

@inline _mul_phase!(X, p) = JACC.parallel_for(length(X)) do i; X[i] *= p; end


# ---------------------------------------------------------------------------
# One-dimensional halo exchange
# ---------------------------------------------------------------------------
function exchange_dim!(ls::Lattice{D,T,TA,T1}, d::Int) where {D,T,TA,T1}
    #println(ls.buf)
    bufM, bufP  = ls.buf[2d-1], ls.buf[2d]
    rankM, rankP = ls.nbr[d]
    myrank      = ls.myrank
    req = MPI.Request[]

    # minus side
    if rankM == myrank
        copy!(_ghost_view(ls.A, ls.nw, d, :minus),
              _face_view(ls.A,  ls.nw, d, :minus))
        if ls.coords[d] == 0
            _mul_phase!(_ghost_view(ls.A, ls.nw, d, :minus), ls.phases[d])
        end
    else
        wrap = (ls.coords[d]==0)
        if wrap
            copy!(bufM, _face_view(ls.A, ls.nw, d, :minus))
            _mul_phase!(bufM, ls.phases[d])
            push!(req, MPI.Isend(bufM, rankM, d, ls.cart))
        else
            push!(req, MPI.Isend(ls.A, rankM, d, ls.cart))
        end
        push!(req, MPI.Irecv!(bufM, rankM, d+D, ls.cart))
    end

    # plus side
    if rankP == myrank
        copy!(_ghost_view(ls.A, ls.nw, d, :plus),
              _face_view(ls.A,  ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d]-1
            _mul_phase!(_ghost_view(ls.A, ls.nw, d, :plus), ls.phases[d])
        end
    else
        wrap = (ls.coords[d]==ls.dims[d]-1)
        if wrap
            copy!(bufP, _face_view(ls.A, ls.nw, d, :plus))
            _mul_phase!(bufP, ls.phases[d])
            push!(req, MPI.Isend(bufP, rankP, d+D, ls.cart))
        else
            push!(req, MPI.Isend(ls.A, rankP, d+D, ls.cart))
        end
        push!(req, MPI.Irecv!(bufP, rankP, d, ls.cart))
    end

    compute_interior!(ls)        # overlap compute
    MPI.Waitall!(req)

    if rankM != myrank
        copy!(_ghost_view(ls.A, ls.nw, d, :minus), bufM)
    end
    if rankP != myrank
        copy!(_ghost_view(ls.A, ls.nw, d, :plus),  bufP)
    end
end

# ---------------------------------------------------------------------------
# --------------- user-replaceable compute kernels ---------------------------
# ---------------------------------------------------------------------------
compute_interior!(ls::Lattice) = nothing      # stencil for inner region
compute_boundary!(ls::Lattice) = nothing      # stencil for ghost region



export Lattice