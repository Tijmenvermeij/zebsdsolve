// Overlap-only normalized SO(3) seed pages. The correlation itself and final
// interpolation remain native SI. All indices retain the native (j,c,a) layout.
struct PeakP {
    n:u32, nc:u32, nb:u32, first:u32,
    rows:u32, core_first:u32, core_rows:u32, grid_x:u32,
    input_count:u32, cursor_set:u32, pad0:u32, pad1:u32,
};
@group(0) @binding(0) var<uniform> pp:PeakP;
@group(0) @binding(1) var<storage,read_write> scores:array<f32>;
@group(0) @binding(2) var<storage,read> reciprocal:array<f32>;
// First 4*nc: (north/south, pole/interior, lambda) maximum and source index.
// Last 2*nc: adjacent-ring minimum and source index, for strict-lower evidence.
@group(0) @binding(3) var<storage,read_write> pole:array<vec2<u32>>;
@group(0) @binding(4) var<storage,read> input_page:array<vec2<u32>>;
@group(0) @binding(5) var<storage,read_write> output_page:array<vec2<u32>>;
@group(0) @binding(6) var<storage,read> cursor:array<vec2<u32>>;
@group(0) @binding(7) var<storage,read> numerator:array<f32>;
const NEG:f32=-3.4e38;
const MISSING:u32=0xffffffffu;
fn finite_score(v:f32)->bool {return v==v && abs(v)<3.4e38;}
fn empty_key()->vec2<u32> {return vec2<u32>(bitcast<u32>(NEG),MISSING);}
fn better(a:vec2<u32>,b:vec2<u32>)->bool {
    let av=bitcast<f32>(a.x);let bv=bitcast<f32>(b.x);
    return av>bv || (av==bv && a.y<b.y);
}
fn global_idx(j:u32,c:u32,a:u32)->u32 {return (j*pp.nc+c)*pp.n+a;}
fn local_idx(j:u32,c:u32,a:u32)->u32 {return ((j-pp.first)*pp.nc+c)*pp.n+a;}
fn lambda_of(side:u32,c:u32,a:u32)->u32 {
    return select((a+c)&(pp.nc-1u),(a+pp.nc-c)&(pp.nc-1u),side==1u);
}
fn key_at(j:u32,c:u32,a:u32)->vec2<u32> {
    if(j==0u || j+1u==pp.nb) {
        let side=select(0u,1u,j>0u);
        return pole[(side*2u)*pp.nc+lambda_of(side,c,a)];
    }
    let v=scores[local_idx(j,c,a)];
    if(!finite_score(v)){return empty_key();}
    return vec2<u32>(bitcast<u32>(v),global_idx(j,c,a));
}
@compute @workgroup_size(256)
fn normalize_slab(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let i=(wid.y*pp.grid_x+wid.x)*256u+tid;
    if(i>=pp.rows*pp.nc*pp.n){return;}
    let r=reciprocal[i];let v=numerator[i]*r;
    scores[i]=select(NEG,v,r>0.0 && finite_score(v));
}
var<workgroup> pv:array<f32,64>;
var<workgroup> pi:array<u32,64>;
var<workgroup> pmin:array<f32,64>;
var<workgroup> pmini:array<u32,64>;
// Collapse physically identical pole aliases. Adjacent ring maxima grouped by
// lambda make the complete quotient neighborhood available without a CPU volume.
@compute @workgroup_size(64)
fn pole_groups(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let group=wid.y*pp.grid_x+wid.x;
    if(group>=4u*pp.nc){return;}
    let lambda=group&(pp.nc-1u);let kind=group/pp.nc;
    let side=kind/2u;let ring=kind&1u;
    let pole_j=select(0u,pp.nb-1u,side==1u);
    let j=select(ring,pp.nb-1u-ring,side==1u);
    if(pole_j<pp.first || pole_j>=pp.first+pp.rows || j<pp.first || j>=pp.first+pp.rows){return;}
    var best=empty_key();
    var lowest=vec2<u32>(bitcast<u32>(3.4e38),MISSING);
    for(var k=tid;k<pp.n;k+=64u) {
        let c=k&(pp.nc-1u);let turn=k/pp.nc;
        let a0=select((lambda+pp.nc-c)&(pp.nc-1u),(lambda+c)&(pp.nc-1u),side==1u);
        let a=a0+turn*pp.nc;let v=scores[local_idx(j,c,a)];
        let key=vec2<u32>(bitcast<u32>(v),global_idx(j,c,a));
        if(finite_score(v)) {
            if(better(key,best)){best=key;}
            let low=bitcast<f32>(lowest.x);
            if(v<low || (v==low && key.y<lowest.y)){lowest=key;}
        }
    }
    pv[tid]=bitcast<f32>(best.x);pi[tid]=best.y;
    pmin[tid]=bitcast<f32>(lowest.x);pmini[tid]=lowest.y;workgroupBarrier();
    for(var off=32u;off>0u;off>>=1u) {
        if(tid<off) {
            let other=vec2<u32>(bitcast<u32>(pv[tid+off]),pi[tid+off]);
            let mine=vec2<u32>(bitcast<u32>(pv[tid]),pi[tid]);
            if(better(other,mine)){pv[tid]=bitcast<f32>(other.x);pi[tid]=other.y;}
            let lo=pmin[tid+off];let li=pmini[tid+off];
            if(lo<pmin[tid] || (lo==pmin[tid] && li<pmini[tid])){pmin[tid]=lo;pmini[tid]=li;}
        }
        workgroupBarrier();
    }
    if(tid==0u){
        pole[group]=vec2<u32>(bitcast<u32>(pv[0]),pi[0]);
        if(ring==1u){pole[(4u+side)*pp.nc+lambda]=vec2<u32>(bitcast<u32>(pmin[0]),pmini[0]);}
    }
    if(ring==0u) {
        for(var k=tid;k<pp.n;k+=64u) {
            let c=k&(pp.nc-1u);let turn=k/pp.nc;
            let a0=select((lambda+pp.nc-c)&(pp.nc-1u),(lambda+c)&(pp.nc-1u),side==1u);
            scores[local_idx(j,c,a0+turn*pp.nc)]=pv[0];
        }
    }
}
fn local_maximum(j:u32,c:u32,a:u32,key:vec2<u32>)->bool {
    let v=bitcast<f32>(key.x);
    if(key.y==MISSING || !finite_score(v)){return false;}
    var lower=false;
    if(j==0u || j+1u==pp.nb) {
        // One representative for each collapsed pole. The neighboring ring and
        // pole groups span lambda +/-2 (all aliases' 26-neighbor union).
        if(c!=0u || a>=pp.nc){return false;}
        let side=select(0u,1u,j>0u);let lambda=lambda_of(side,c,a);
        for(var ring=0u;ring<2u;ring+=1u) {
            for(var d=-2;d<=2;d+=1) {
                if(ring==0u && d==0){continue;}
                let l=u32(i32(lambda)+i32(pp.nc)+d)&(pp.nc-1u);
                let other=pole[(side*2u+ring)*pp.nc+l];
                if(other.y==MISSING){continue;}
                if(better(other,key)){return false;}
                lower=lower || bitcast<f32>(other.x)<v;
                if(ring==1u){let low=pole[(4u+side)*pp.nc+l];lower=lower || (low.y!=MISSING && bitcast<f32>(low.x)<v);}
            }
        }
    } else {
        for(var dj=-1;dj<=1;dj+=1) {for(var dc=-1;dc<=1;dc+=1) {for(var da=-1;da<=1;da+=1) {
            if(dj==0 && dc==0 && da==0){continue;}
            let jj=u32(i32(j)+dj);
            let cc=u32(i32(c)+i32(pp.nc)+dc)&(pp.nc-1u);
            let aa=u32(i32(a)+i32(pp.n)+da)&(pp.n-1u);
            let other=key_at(jj,cc,aa);
            if(other.y==MISSING){continue;}
            if(better(other,key)){return false;}
            lower=lower || bitcast<f32>(other.x)<v;
        }}}
    }
    // Defined local lexicographic convention: require at least one strictly
    // lower valid neighbor. Constant/isolated undefined fields yield no extras.
    return lower;
}
var<workgroup> keys:array<vec2<u32>,256>;
fn sort_keys(tid:u32) {
    for(var width=2u;width<=256u;width*=2u) {
        for(var gap=width/2u;gap>0u;gap>>=1u) {
            let mine=keys[tid];let other=keys[tid^gap];
            let first=((tid&width)==0u)==((tid&gap)==0u);
            let take=select(better(mine,other),better(other,mine),first);
            workgroupBarrier();
            keys[tid]=select(mine,other,take);
            workgroupBarrier();
        }
    }
}
@compute @workgroup_size(256)
fn block_maxima(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let block=wid.y*pp.grid_x+wid.x;let i=block*256u+tid;
    var key=empty_key();
    if(i<pp.core_rows*pp.nc*pp.n) {
        let a=i&(pp.n-1u);let c=(i/pp.n)&(pp.nc-1u);
        let j=pp.core_first+i/(pp.n*pp.nc);let proposed=key_at(j,c,a);
        var after=true;
        if(pp.cursor_set!=0u) {
            let v=bitcast<f32>(proposed.x);let cv=bitcast<f32>(cursor[0].x);
            after=v<cv || (v==cv && proposed.y>cursor[0].y);
        }
        if(after && local_maximum(j,c,a,proposed)){key=proposed;}
    }
    keys[tid]=key;workgroupBarrier();sort_keys(tid);
    if(tid<8u){output_page[block*8u+tid]=keys[tid];}
}
@compute @workgroup_size(256)
fn reduce_pages(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) tid:u32) {
    let block=wid.y*pp.grid_x+wid.x;let i=block*256u+tid;
    var key=empty_key();if(i<pp.input_count){key=input_page[i];}
    keys[tid]=key;workgroupBarrier();sort_keys(tid);
    if(tid<8u){output_page[block*8u+tid]=keys[tid];}
}
