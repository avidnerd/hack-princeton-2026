import sys, trimesh, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import Polygon
P=trimesh.load("ref/Wrist_Roll_Pitch_SO101.stl")
zs=[float(z) for z in sys.argv[1].split(",")]; out=sys.argv[2]
fig,axs=plt.subplots(2,(len(zs)+1)//2,figsize=(4.6*((len(zs)+1)//2),9))
for ax,z in zip(axs.flat,zs):
    s=P.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    ax.set_title(f"Z = {z}"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.add_patch(plt.Rectangle((-10.2,-12.4),45.4,24.8,fill=False,ec="r",ls="--",lw=1))
    ax.plot(0,0,"k+",ms=12); ax.plot(-1.8,0,"bx",ms=8)
    for x,y in [(28.4,10),(28.4,-10),(8.5,10),(8.5,-10)]: ax.plot(x,y,"g.",ms=8)
    ax.set_xlim(-30,38); ax.set_ylim(-22,18)
    if s is None: continue
    p,_=s.to_2D(to_2D=np.eye(4))
    print(f"Z={z}")
    for g in p.polygons_full:
        xs,ys=g.exterior.xy; ax.fill(xs,ys,alpha=.35,color="C0"); ax.plot(xs,ys,"C0",lw=.8)
        print("   region bounds", np.round(g.bounds,2).tolist())
        for h in g.interiors:
            hx,hy=h.xy; ax.fill(hx,hy,color="w"); ax.plot(hx,hy,"C3",lw=.8)
            hp=Polygon(h); c=hp.centroid; d=2*np.sqrt(hp.area/np.pi)
            print(f"   hole at ({c.x:.2f},{c.y:.2f}) d_eq={d:.2f} bounds {np.round(hp.bounds,2).tolist()}")
fig.suptitle("Wrist_Roll_Pitch_SO101 sections. red dashed = servo footprint; + = URDF wrist-roll axis; x = brief axis; green = brief screw holes")
fig.tight_layout(); fig.savefig(out,dpi=110)
