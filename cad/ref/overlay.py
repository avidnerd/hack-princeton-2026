import sys, trimesh, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from trimesh.transformations import euler_matrix, translation_matrix, inverse_matrix
def T(xyz,rpy): return translation_matrix(xyz) @ euler_matrix(*rpy, axes="sxyz")
Tps=inverse_matrix(T([0,-0.028,0.0181],[-1.5708,-1.5708,0]))@T([0,-0.0424,0.0306],[1.5708,1.5708,0])
V=trimesh.load("ref/sts3215_03a_no_horn_v1.stl"); V.apply_transform(Tps); V.apply_scale(1000); V.export("ref/servo_in_part_frame.stl")
P=trimesh.load("ref/Wrist_Roll_Pitch_SO101.stl")
zs=[float(z) for z in sys.argv[1].split(",")]
def draw(ax,m,z,c,lab):
    s=m.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    if s is None: return []
    p,_=s.to_2D(to_2D=np.eye(4)); out=[]
    for g in p.polygons_full:
        xs,ys=g.exterior.xy; ax.fill(xs,ys,alpha=.3,color=c); ax.plot(xs,ys,color=c,lw=.8)
        for h in g.interiors: hx,hy=h.xy; ax.plot(hx,hy,color=c,lw=.8,ls=":")
        out.append(g)
    return out
fig,axs=plt.subplots(2,(len(zs)+1)//2,figsize=(4.6*((len(zs)+1)//2),9))
for ax,z in zip(axs.flat,zs):
    gp=draw(ax,P,z,"C0","part"); gs=draw(ax,V,z,"C3","servo")
    inter=sum(a.intersection(b).area for a in gp for b in gs)
    ax.set_title(f"Z={z}  overlap {inter:.1f} mm²"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.plot(0,0,"k+",ms=12); ax.set_xlim(-30,38); ax.set_ylim(-22,18)
    sb=[np.round(g.bounds,1).tolist() for g in gs]; print(f"Z={z} servo regions {sb} overlap {inter:.2f}")
fig.suptitle("blue = Wrist_Roll_Pitch part, red = STS3215 (URDF pose). + = wrist-roll axis")
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=110)
print("servo z-range", np.round(V.bounds[:,2],2))
