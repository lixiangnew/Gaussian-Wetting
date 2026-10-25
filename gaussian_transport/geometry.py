"""Explicit clipped Voronoi control volumes for reliable M3 route A.

2D domains are unions of non-overlapping convex pieces with an explicit outer
boundary. 3D currently supports a convex axis-aligned box through a small
NumPy half-space reference implementation. Lengths are metres, volumes m^d times measure_scale,
and face areas m^(d-1) times measure_scale.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


def rectangle(x0, y0, x1, y1):
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], float)


@dataclass(frozen=True)
class Domain2D:
    pieces: tuple
    boundary: np.ndarray
    thickness: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'pieces', tuple(np.asarray(p, float) for p in self.pieces))
        object.__setattr__(self, 'boundary', np.asarray(self.boundary, float))
        if self.thickness <= 0 or not np.isfinite(self.thickness):
            raise ValueError('positive finite thickness required')
        for p in self.pieces + (self.boundary,):
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
                raise ValueError('valid 2D polygons required')
            if polygon_area_centroid(p)[0] <= 0:
                raise ValueError('polygons must be counter-clockwise')

    @property
    def volume(self):
        return sum(polygon_area_centroid(p)[0] for p in self.pieces)*self.thickness


def box_domain(width=1.0, height=1.0, thickness=1.0):
    p = rectangle(0,0,width,height)
    return Domain2D((p,), p, thickness)


def l_shape_domain(thickness=1.0):
    boundary = np.array([[0,0],[1,0],[1,.4],[.4,.4],[.4,1],[0,1]], float)
    pieces = (rectangle(0,0,1,.4), rectangle(0,.4,.4,1))
    return Domain2D(pieces, boundary, thickness)


def polygon_area_centroid(poly):
    p = np.asarray(poly, float); q = np.roll(p, -1, axis=0)
    cross = p[:,0]*q[:,1]-q[:,0]*p[:,1]
    signed = .5*cross.sum()
    if abs(signed) < 1e-16:
        return 0.0, p.mean(axis=0)
    centroid = np.array([np.sum((p[:,0]+q[:,0])*cross),
                         np.sum((p[:,1]+q[:,1])*cross)])/(6*signed)
    return signed, centroid


def clip_polygon(poly, a, b, tol=1e-12):
    """Clip a CCW polygon by a.x <= b (Sutherland--Hodgman)."""
    out=[]; poly=np.asarray(poly,float)
    for s,e in zip(poly, np.roll(poly,-1,axis=0)):
        fs=float(a@s-b); fe=float(a@e-b); ins=fs<=tol; ine=fe<=tol
        if ins: out.append(s)
        if ins != ine:
            out.append(s+(e-s)*(fs/(fs-fe)))
    return np.asarray(out,float) if out else np.empty((0,2))


def _piece_constraints(poly):
    result=[]
    for p,q in zip(poly,np.roll(poly,-1,axis=0)):
        edge=q-p; a=np.array([edge[1],-edge[0]])
        result.append((a,float(a@p)))
    return result


def _line_interval(point, direction, constraints, lo=-np.inf, hi=np.inf, tol=1e-12):
    for a,b in constraints:
        den=float(a@direction); rhs=float(b-a@point)
        if abs(den)<tol:
            if rhs < -tol: return None
        elif den>0: hi=min(hi,rhs/den)
        else: lo=max(lo,rhs/den)
        if lo>hi+tol: return None
    return lo,hi


def _merge_intervals(intervals, tol=1e-10):
    if not intervals: return []
    result=[]
    for lo,hi in sorted(intervals):
        if hi-lo<=tol: continue
        if result and lo<=result[-1][1]+tol: result[-1]=(result[-1][0],max(hi,result[-1][1]))
        else: result.append((lo,hi))
    return result


@dataclass
class GraphGeometry:
    centers: np.ndarray
    volume: np.ndarray
    centroid: np.ndarray
    edges: np.ndarray
    face_area: np.ndarray
    face_normal: np.ndarray
    face_center: np.ndarray
    center_distance: np.ndarray
    boundary_node: np.ndarray
    boundary_area: np.ndarray
    boundary_normal: np.ndarray
    boundary_center: np.ndarray
    method: str

    def validate(self):
        n=len(self.centers); e=len(self.edges)
        if self.volume.shape!=(n,) or np.any(self.volume<=0): raise ValueError('positive volumes required')
        if self.edges.shape!=(e,2) or np.any(self.edges[:,0]>=self.edges[:,1]): raise ValueError('canonical edges required')
        if len(np.unique(self.edges,axis=0))!=e: raise ValueError('duplicate edges')
        if self.face_area.shape!=(e,) or np.any(self.face_area<=0): raise ValueError('positive faces required')
        if np.any(self.center_distance<=0): raise ValueError('positive distances required')
        return self


def disjoint_union(*geometries):
    """Join already validated components without inventing cross-component edges."""
    if not geometries: raise ValueError('at least one geometry required')
    dim=geometries[0].centers.shape[1]; offsets=np.cumsum([0]+[len(g.centers) for g in geometries[:-1]])
    if any(g.centers.shape[1]!=dim for g in geometries): raise ValueError('dimension mismatch')
    def cat(name): return np.concatenate([getattr(g,name) for g in geometries],axis=0)
    edges=np.concatenate([g.edges+off for g,off in zip(geometries,offsets)],axis=0)
    bn=np.concatenate([g.boundary_node+off for g,off in zip(geometries,offsets)])
    return GraphGeometry(cat('centers'),cat('volume'),cat('centroid'),edges,cat('face_area'),cat('face_normal'),
      cat('face_center'),cat('center_distance'),bn,cat('boundary_area'),cat('boundary_normal'),cat('boundary_center'),
      'disjoint_union').validate()


def voronoi_2d(centers, domain):
    """Clipped Voronoi geometry on a box or convex-piece union."""
    x=np.asarray(centers,float); n=len(x)
    if x.shape!=(n,2) or n<1 or not np.isfinite(x).all(): raise ValueError('finite [N,2] centers required')
    # Strict inside test via piece clipping membership.
    for point in x:
        if not any(all(a@point<=b+1e-10 for a,b in _piece_constraints(p)) for p in domain.pieces):
            raise ValueError('center outside domain')
    site_constraints=[]
    for i in range(n):
        cs=[]
        for j in range(n):
            if i==j: continue
            a=2*(x[j]-x[i]); b=float(x[j]@x[j]-x[i]@x[i]); cs.append((a,b))
        site_constraints.append(cs)
    volume=np.zeros(n); centroid=np.zeros_like(x)
    for i in range(n):
        moment=np.zeros(2)
        for piece in domain.pieces:
            poly=piece.copy()
            for a,b in site_constraints[i]:
                poly=clip_polygon(poly,a,b)
                if not len(poly): break
            if len(poly):
                area,c=polygon_area_centroid(poly); volume[i]+=area; moment+=area*c
        if volume[i]<=1e-14: raise ValueError('empty clipped Voronoi cell')
        centroid[i]=moment/volume[i]
    volume*=domain.thickness
    edge_rows=[]
    for i in range(n):
        for j in range(i+1,n):
            a=x[j]-x[i]; b=.5*float(x[j]@x[j]-x[i]@x[i])
            norm=np.linalg.norm(a); normal=a/norm; tangent=np.array([-normal[1],normal[0]])
            point=a*(b/(a@a)); intervals=[]
            for piece in domain.pieces:
                constraints=_piece_constraints(piece)+site_constraints[i]
                val=_line_interval(point,tangent,constraints)
                if val is not None: intervals.append(val)
            merged=_merge_intervals(intervals)
            length=sum(hi-lo for lo,hi in merged)
            if length>1e-10:
                midpoint=sum((hi-lo)*(point+.5*(lo+hi)*tangent) for lo,hi in merged)/length
                edge_rows.append((i,j,length*domain.thickness,normal,midpoint,norm))
    b_rows=[]
    boundary=domain.boundary
    for p,q in zip(boundary,np.roll(boundary,-1,axis=0)):
        vector=q-p; length=np.linalg.norm(vector); direction=vector/length
        outward=np.array([direction[1],-direction[0]])
        for i in range(n):
            val=_line_interval(p,vector,site_constraints[i],0.,1.)
            if val is not None and val[1]-val[0]>1e-10:
                lo,hi=val; seglen=(hi-lo)*length
                b_rows.append((i,seglen*domain.thickness,outward,p+.5*(lo+hi)*vector))
    edges=np.array([[r[0],r[1]] for r in edge_rows],int).reshape(-1,2)
    g=GraphGeometry(x,volume,centroid,edges,
        np.array([r[2] for r in edge_rows]),np.array([r[3] for r in edge_rows]).reshape(-1,2),
        np.array([r[4] for r in edge_rows]).reshape(-1,2),np.array([r[5] for r in edge_rows]),
        np.array([r[0] for r in b_rows],int),np.array([r[1] for r in b_rows]),
        np.array([r[2] for r in b_rows]).reshape(-1,2),np.array([r[3] for r in b_rows]).reshape(-1,2),
        'clipped_voronoi_2d')
    return g.validate()


def _polygon_on_plane(points, normal, tol=1e-10):
    """Area and centroid of a convex planar polygon from unordered vertices."""
    n=normal/np.linalg.norm(normal)
    seed=np.array([1.,0.,0.]) if abs(n[0])<.8 else np.array([0.,1.,0.])
    u=np.cross(n,seed); u/=np.linalg.norm(u); v=np.cross(n,u)
    origin=points.mean(axis=0); uv=np.column_stack(((points-origin)@u,(points-origin)@v))
    order=np.argsort(np.arctan2(uv[:,1],uv[:,0])); p=points[order]; q=np.roll(p,-1,axis=0)
    tri=np.cross(p-origin,q-origin); weights=.5*(tri@n)
    keep=weights>tol
    if not np.any(keep): return 0.,origin
    area=float(weights[keep].sum())
    centroid=np.sum(weights[keep,None]*(origin+p[keep]+q[keep])/3.,axis=0)/area
    return area,centroid


def _polyhedron_from_halfspaces(A, b, tol=2e-9):
    """Intersect `A x <= b` in 3D by enumerating plane triples.

    This is deliberately a fixed small-problem reference: it is deterministic
    and dependency-free, but not intended as a large 3D production hull code.
    """
    A=np.asarray(A,float); b=np.asarray(b,float); vertices=[]
    scale=max(1.,float(np.max(np.abs(b))))
    for ids in combinations(range(len(A)),3):
        M=A[list(ids)]; det=np.linalg.det(M)
        if abs(det)<1e-12: continue
        point=np.linalg.solve(M,b[list(ids)])
        if np.all(A@point<=b+tol*scale): vertices.append(point)
    if not vertices: raise ValueError('empty or degenerate clipped 3D cell')
    vertices=np.unique(np.round(np.asarray(vertices),12),axis=0)
    faces={}; volume=0.
    for k,(a,rhs) in enumerate(zip(A,b)):
        pts=vertices[np.abs(vertices@a-rhs)<=20*tol*max(1.,np.linalg.norm(a),abs(rhs))]
        if len(pts)<3: continue
        area,center=_polygon_on_plane(pts,a,tol=tol*tol)
        if area<=tol: continue
        normal=a/np.linalg.norm(a); faces[k]=(area,center,normal)
        volume+=area*float(normal@center)/3.
    if volume<=tol: raise ValueError('invalid clipped 3D volume')
    return vertices,volume,vertices.mean(axis=0),faces


def voronoi_box_3d(centers, lower=(0,0,0), upper=(1,1,1), tol=2e-8):
    """Small fixed 3D clipped Voronoi reference using NumPy halfspaces."""
    x=np.asarray(centers,float); lo=np.asarray(lower,float); hi=np.asarray(upper,float); n=len(x)
    if x.shape!=(n,3) or np.any(x<=lo) or np.any(x>=hi): raise ValueError('strictly interior [N,3] centers required')
    cells=[]; volume=np.zeros(n); centroid=np.zeros_like(x); face_maps=[]
    for i in range(n):
        A=[]; b=[]; keys=[]
        for d in range(3):
            a=np.zeros(3); a[d]=-1; A.append(a); b.append(-lo[d]); keys.append(('box',2*d))
            a=np.zeros(3); a[d]=1; A.append(a); b.append(hi[d]); keys.append(('box',2*d+1))
        for j in range(n):
            if i==j: continue
            a=x[j]-x[i]; A.append(a); b.append(.5*(x[j]@x[j]-x[i]@x[i])); keys.append(('site',j))
        vertices,volume[i],centroid[i],indexed=_polyhedron_from_halfspaces(A,b,tol)
        cells.append(vertices); face_maps.append({keys[k]:value for k,value in indexed.items()})
    rows=[]
    for i in range(n):
        for j in range(i+1,n):
            delta=x[j]-x[i]; dist=np.linalg.norm(delta); normal=delta/dist
            face=face_maps[i].get(('site',j))
            if face is not None and face[0]>tol: rows.append((i,j,face[0],normal,face[1],dist))
    edges=np.array([[r[0],r[1]] for r in rows],int).reshape(-1,2)
    # Boundary patches omitted in this closed-domain M6 reference.
    g=GraphGeometry(x,volume,centroid,edges,np.array([r[2] for r in rows]),
      np.array([r[3] for r in rows]).reshape(-1,3),np.array([r[4] for r in rows]).reshape(-1,3),
      np.array([r[5] for r in rows]),np.empty(0,int),np.empty(0),np.empty((0,3)),np.empty((0,3)),
      'clipped_voronoi_box_3d')
    return g.validate()
