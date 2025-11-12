# dna_fasta_holo_cell.py
# --------------------------------------------------------------
# Holographic volumetric FASTA-driven φ-spiral cell with division
# Fully 3D, strands, organelles, backpressure, echoes, no flat pancake
# Install: pip install vispy pyqt6 numpy
# Run: python dna_fasta_holo_cell.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)
core_radius = 12.0
strand_sep = 0.7
twist_factor = 4*np.pi
MAX_HISTORY = 8000
DIVISION_INTERVAL = 2000

# nucleotide → geometry mapping
base_map = {'A':5,'T':2,'G':4,'C':1}

# geometries: (dim, note, color, name, alpha, vertices)
geometries = [
    (1, 'C', 'red', 'Point',0.015269,1),
    (2, 'D', 'green','Line',0.008262,2),
    (3, 'E', 'violet','Triangle',0.110649,3),
    (4, 'F', 'mediumpurple','Tetrahedron',-0.083485,4),
    (5, 'G', 'blue','Pentachoron',0.025847,5),
    (6, 'A', 'indigo','Hexacross',-0.045123,12),
    (7, 'B', 'purple','Heptacube',0.067891,14),
    (8, 'C', 'white','Octacube',0.012345,16)
]
angles = [i*golden_angle_deg for i in range(8)]

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq=[]
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path="ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: "+fasta_path)

genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400,900), bgcolor='black',
                           title="Holographic FASTA Cell")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- ORGANELLE & LATTICE HELPERS ----------
def spawn_organelle(center,color,n=12,size=0.5):
    pts=center+np.random.normal(scale=0.2,size=(n,3))*size
    rgba=list(Color(color).rgba); rgba[3]=1.0
    mark=Markers(pos=pts,face_color=rgba,edge_color=None,size=5,parent=view.scene)
    return {"marker":mark,"positions":pts,"color":rgba,"size":size}

def lattice_push(pt,nodes,positive=True,strength=0.02):
    if len(nodes)==0: return pt
    nearest=nodes[np.random.randint(0,len(nodes))]
    vec=nearest-pt
    if not positive: vec*=-1
    return pt+vec*strength

# ---------- CELL CLASS ----------
class HoloCell:
    def __init__(self,center_offset=np.zeros(3)):
        self.offset=center_offset
        self.frame=0
        self.strand1=Line(pos=np.zeros((1,3)),color=(0,1,1,0.9),width=2,parent=view.scene)
        self.strand2=Line(pos=np.zeros((1,3)),color=(1,0.5,0,0.9),width=2,parent=view.scene)
        self.rungs,self.echoes,self.links,self.labels,self.centers,self.organelles=[],[],[],[],[],[]
        self.progress_text=Text("0%",pos=self.offset+[0,0,20],color='white',
                                font_size=24,anchor_x='center',parent=view.scene)

    def update(self):
        self.frame+=1
        N=500
        t=np.linspace(0,self.frame,N)
        s1,s2=[],[]

        for tt in t:
            idx=int(tt)%genome_len
            base=genome_seq[idx]
            dim=base_map.get(base,1)-1
            _,_,col,_,_,verts=geometries[dim]

            # 3D volumetric φ-spiral (no pancake)
            r=core_radius*(1-(tt/genome_len)**1.5)
            r=max(r,0.5)
            theta=np.radians(golden_angle_deg)*tt
            twist=tt/genome_len*twist_factor
            z=np.sin(tt/genome_len*np.pi*4)*3 + (tt/genome_len)*10

            a1=np.radians(angles[dim])
            s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
            a2=np.radians(-angles[dim])
            s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])

        s1,s2=np.array(s1)+self.offset,np.array(s2)+self.offset
        self.strand1.set_data(s1)
        self.strand2.set_data(s2)

        # Generate rungs, echoes, organelles
        idx_frame=self.frame%genome_len
        base=genome_seq[idx_frame]
        dim=base_map.get(base,1)-1
        _,_,col,name,_,verts=geometries[dim]

        if self.frame%20==0:
            step=max(1,len(s1)//verts)
            idx_pts=np.arange(0,verts*step,step)[:verts]
            pts1=s1[idx_pts]; pts2=s2[idx_pts]; all_pts=np.vstack((pts1,pts2))
            mark=Markers(pos=all_pts,face_color=Color(col).rgba,
                         edge_color='white',size=6,parent=view.scene)
            self.rungs.append(mark)
            cen=all_pts.mean(axis=0); self.centers.append(cen)
            lbl=Text(f"{base}:{name}",pos=cen+[0,0,0.3],color=col,font_size=10,
                     bold=True,anchor_x='center',parent=view.scene)
            self.labels.append(lbl)

            if len(self.centers)>1:
                echo_pts=all_pts*0.75+np.random.normal(scale=0.01,size=all_pts.shape)
                echo=Markers(pos=echo_pts,face_color=(1,1,1,0.25),size=4,parent=view.scene)
                self.echoes.append(echo)

            if len(self.centers)>1:
                prev_c=self.centers[-2]; segs=[]
                for p in all_pts[:6]: segs+=[prev_c,p]
                link=Line(pos=np.array(segs),color=(0.7,0.7,0.7,0.4),
                          width=1,connect='segments',parent=view.scene)
                self.links.append(link)

            # Organelles
            spawn_prob=0.02+dim*0.02; cluster_size=6+dim
            if np.random.rand()<spawn_prob:
                org=spawn_organelle(cen,col,n=cluster_size,size=0.3)
                self.organelles.append(org)

        # Lattice backpressure
        lattice_nodes=np.array(self.centers)
        for org in self.organelles:
            positive=base in ['A','T']
            new_pts=np.array([lattice_push(p,lattice_nodes,positive=positive,strength=0.02)
                              for p in org['positions']])
            org['positions']=new_pts; org['marker'].set_data(pos=new_pts,face_color=org['color'],size=5)

        # Camera rotation
        view.camera.azimuth=self.frame*0.3
        view.camera.elevation=25+5*np.sin(self.frame*0.005)

        # Progress
        percent=min(self.frame/genome_len*100,100)
        self.progress_text.text=f"{percent:.2f}%"

        # History limit
        if len(self.rungs)>MAX_HISTORY:
            drop=len(self.rungs)//3
            self.rungs=self.rungs[drop:]; self.echoes=self.echoes[drop:]
            self.links=self.links[drop:]; self.labels=self.labels[drop:]
            self.centers=self.centers[drop:]; self.organelles=self.organelles[drop:]

# ---------- INIT ----------
cells=[HoloCell(center_offset=np.array([0,0,0]))]

def update(ev):
    global cells
    new_cells=[]
    for c in cells:
        c.update()
        if c.frame % DIVISION_INTERVAL==0 and c.frame>0:
            offset=np.random.normal(scale=6.0,size=3)
            new_cell=HoloCell(center_offset=c.offset+offset)
            new_cells.append(new_cell)
    cells+=new_cells

# ---------- START ----------
timer=app.Timer(interval=0.02,connect=update,start=True)

if __name__=="__main__":
    canvas.show()
    app.run()
