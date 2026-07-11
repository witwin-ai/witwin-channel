#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr float kEps = 1.0e-6f;
constexpr float kHalfPiMinusOffset = 1.52079632679f;

struct Vec3 { float x, y, z; };

__device__ __forceinline__ Vec3 v3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) { return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) { return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ __forceinline__ Vec3 mul(Vec3 a, float s) { return v3(a.x*s,a.y*s,a.z*s); }
__device__ __forceinline__ float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float norm(Vec3 a) { return sqrtf(fmaxf(dot(a,a),0.f)); }
__device__ __forceinline__ Vec3 normalize(Vec3 a, Vec3 default_value=v3(0.f,0.f,1.f)) {
    const float n=norm(a); return n>kEps?mul(a,1.f/n):default_value;
}
__device__ __forceinline__ Vec3 load3(const float *p,int i) { return v3(p[i*3],p[i*3+1],p[i*3+2]); }

__device__ __forceinline__ Vec3 silhouette_viewpoint(
    Vec3 hit_p, Vec3 shading_n, Vec3 geometric_n, Vec3 ray_dir) {
    Vec3 geo = norm(geometric_n)>kEps ? geometric_n : shading_n;
    // Mitsuba's primitive_silhouette_projection uses the unmodified
    // geometric interaction normal (si.n), not a face-forward normal.
    Vec3 surface_n = geo;
    Vec3 tangent = sub(ray_dir,mul(surface_n,dot(ray_dir,surface_n)));
    if (norm(tangent)<=kEps) {
        Vec3 fx=cross(surface_n,v3(1.f,0.f,0.f));
        Vec3 fy=cross(surface_n,v3(0.f,1.f,0.f));
        tangent=norm(fx)>kEps?fx:fy;
    }
    tangent=normalize(tangent,v3(1.f,0.f,0.f));
    Vec3 d=add(mul(surface_n,cosf(kHalfPiMinusOffset)),mul(tangent,sinf(kHalfPiMinusOffset)));
    // Sionna's primitive_silhouette_projection uses a fixed 0.1 scene-unit
    // viewpoint displacement.
    const float offset=0.1f;
    return add(hit_p,mul(d,offset));
}

__device__ __forceinline__ bool wedge_exterior(Vec3 from_edge,Vec3 edge_dir,Vec3 n0,Vec3 n1) {
    Vec3 eh=normalize(edge_dir);
    Vec3 projected=sub(from_edge,mul(eh,dot(from_edge,eh)));
    return norm(projected)>kEps && (dot(projected,n0)>=-kEps || dot(projected,n1)>=-kEps);
}

__device__ int sampled_edge(
    Vec3 tx, Vec3 ray_dir, Vec3 hit_p, Vec3 hit_n, Vec3 hit_geo_n, int prim,
    int sample_index,
    const int *tri_count,const int *tri_edges,int slots,int tri_n,
    const float *edge_pos,const float *edge_dir,const float *n0p,const float *n1p,
    const float *tminp,const float *tmaxp,const int *face1,int edge_n) {
    if(prim<0||prim>=tri_n) return -1;
    const int count=min(max(tri_count[prim],0),slots);
    Vec3 viewpoint=silhouette_viewpoint(hit_p,hit_n,hit_geo_n,ray_dir);
    int valid_count=0;
    for(int s=0;s<count;++s){
        int e=tri_edges[prim*slots+s]; if(e<0||e>=edge_n) continue;
        Vec3 ep=load3(edge_pos,e), ed=load3(edge_dir,e), eh=normalize(ed);
        float ell=fminf(fmaxf(dot(sub(viewpoint,ep),eh),tminp[e]),tmaxp[e]);
        Vec3 point=add(ep,mul(eh,ell));
        Vec3 n0=load3(n0p,e),n1=load3(n1p,e);
        bool flip=dot(ray_dir,n0)>0.f;
        if(!wedge_exterior(sub(tx,point),ed,flip?n1:n0,flip?n0:n1)) continue;
        ++valid_count;
    }
    if(valid_count<=0) return -1;
    unsigned int h=static_cast<unsigned int>(sample_index)^0x9e3779b9u;
    h^=h>>16; h*=0x7feb352du; h^=h>>15; h*=0x846ca68bu; h^=h>>16;
    const int wanted=static_cast<int>(h%static_cast<unsigned int>(valid_count));
    int ordinal=0;
    for(int s=0;s<count;++s){
        int e=tri_edges[prim*slots+s]; if(e<0||e>=edge_n) continue;
        Vec3 ep=load3(edge_pos,e), ed=load3(edge_dir,e), eh=normalize(ed);
        float ell=fminf(fmaxf(dot(sub(viewpoint,ep),eh),tminp[e]),tmaxp[e]);
        Vec3 point=add(ep,mul(eh,ell)); Vec3 en0=load3(n0p,e),en1=load3(n1p,e);
        bool flip=dot(ray_dir,en0)>0.f;
        if(!wedge_exterior(sub(tx,point),ed,flip?en1:en0,flip?en0:en1)) continue;
        if(ordinal++==wanted) return e;
    }
    return -1;
}

__global__ void discover_kernel(
    const float *tx,const float *ray_dir,const int *prim,const float *hit_p,
    const float *hit_n,const float *hit_geo_n,const int *hit_count,int capacity,
    const int *tri_count,const int *tri_edges,int slots,int tri_n,
    const float *edge_pos,const float *edge_dir,const float *n0,const float *n1,
    const float *tmin,const float *tmax,const int *face1,int edge_n,int *seen) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int n=hit_count?min(max(hit_count[0],0),capacity):capacity;
    if(i>=n) return;
    // Discovery stores support, not a per-path contribution. Select one
    // exterior primitive-perimeter candidate per first hit with a reproducible
    // uniform draw; the subsequent estimator samples edge length explicitly.
    int e=sampled_edge(load3(tx,0),load3(ray_dir,i),load3(hit_p,i),load3(hit_n,i),
        load3(hit_geo_n,i),prim[i],i,tri_count,tri_edges,slots,tri_n,edge_pos,edge_dir,
        n0,n1,tmin,tmax,face1,edge_n);
    if(e>=0) atomicExch(seen+e,1);
}

const at::Tensor &required(const at::Tensor *t,const char *name){
    if(!t) throw std::runtime_error(std::string("channel diffraction discovery received null ")+name);
    return *t;
}

void discover(
    const at::Tensor *tx,const at::Tensor *ray_dir,const at::Tensor *prim,
    const at::Tensor *hit_p,const at::Tensor *hit_n,const at::Tensor *hit_geo_n,
    const at::Tensor *hit_count,const at::Tensor *tri_count,const at::Tensor *tri_edges,
    const at::Tensor *edge_pos,const at::Tensor *edge_dir,const at::Tensor *n0,
    const at::Tensor *n1,const at::Tensor *tmin,const at::Tensor *tmax,
    const at::Tensor *face1,at::Tensor *out){
    if(!out) throw std::runtime_error("channel diffraction discovery received null output");
    const auto &rd=required(ray_dir,"ray_dir"); const auto &ep=required(edge_pos,"edge_pos");
    int64_t capacity=rd.size(0),edges=ep.size(0);
    at::Tensor seen=at::empty({edges},ep.options().dtype(at::kInt));
    if(edges>0){
        C10_CUDA_CHECK(cudaMemsetAsync(
            seen.data_ptr<int>(), 0, static_cast<size_t>(edges) * sizeof(int),
            at::cuda::getCurrentCUDAStream()));
    }
    if(capacity>0&&edges>0){
        int blocks=static_cast<int>((capacity+kBlockSize-1)/kBlockSize);
        const auto &te=required(tri_edges,"triangle_edge_indices");
        discover_kernel<<<blocks,kBlockSize,0,at::cuda::getCurrentCUDAStream()>>>(
            required(tx,"tx_pos").data_ptr<float>(),rd.data_ptr<float>(),required(prim,"prim_index").data_ptr<int>(),
            required(hit_p,"hit_p").data_ptr<float>(),required(hit_n,"hit_n").data_ptr<float>(),
            required(hit_geo_n,"hit_geo_n").data_ptr<float>(),hit_count?hit_count->data_ptr<int>():nullptr,
            static_cast<int>(capacity),required(tri_count,"triangle_edge_count").data_ptr<int>(),te.data_ptr<int>(),
            static_cast<int>(te.size(1)),static_cast<int>(required(tri_count,"triangle_edge_count").size(0)),
            ep.data_ptr<float>(),required(edge_dir,"edge_dir").data_ptr<float>(),required(n0,"edge_n0").data_ptr<float>(),
            required(n1,"edge_n1").data_ptr<float>(),required(tmin,"edge_t_min").data_ptr<float>(),
            required(tmax,"edge_t_max").data_ptr<float>(),required(face1,"edge_face1").data_ptr<int>(),
            static_cast<int>(edges),seen.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    *out=at::nonzero(seen).reshape({-1}).to(at::kInt).contiguous();
}

} // namespace

extern "C" void channel_native_diffraction_discover_edges(
    const at::Tensor *tx,const at::Tensor *ray_dir,const at::Tensor *prim,
    const at::Tensor *hit_p,const at::Tensor *hit_n,const at::Tensor *hit_geo_n,
    const at::Tensor *tri_count,const at::Tensor *tri_edges,const at::Tensor *edge_pos,
    const at::Tensor *edge_dir,const at::Tensor *n0,const at::Tensor *n1,
    const at::Tensor *tmin,const at::Tensor *tmax,const at::Tensor *face1,at::Tensor *out){
    discover(tx,ray_dir,prim,hit_p,hit_n,hit_geo_n,nullptr,tri_count,tri_edges,edge_pos,edge_dir,n0,n1,tmin,tmax,face1,out);
}

extern "C" void channel_native_diffraction_discover_edges_counted(
    const at::Tensor *tx,const at::Tensor *ray_dir,const at::Tensor *prim,
    const at::Tensor *hit_p,const at::Tensor *hit_n,const at::Tensor *hit_geo_n,
    const at::Tensor *hit_count,const at::Tensor *tri_count,const at::Tensor *tri_edges,
    const at::Tensor *edge_pos,const at::Tensor *edge_dir,const at::Tensor *n0,
    const at::Tensor *n1,const at::Tensor *tmin,const at::Tensor *tmax,
    const at::Tensor *face1,at::Tensor *out){
    discover(tx,ray_dir,prim,hit_p,hit_n,hit_geo_n,hit_count,tri_count,tri_edges,edge_pos,edge_dir,n0,n1,tmin,tmax,face1,out);
}
