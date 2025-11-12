"""
FASTA-Driven φ-Harmonic Spiral with Recursive φ-Echo Lattice
-------------------------------------------------------------
Everything is genome-driven; strands, rungs, organelles, echoes.
Soft spherical vessel confines the lattice.
Run: python fasta_phi_recursive_vessel.py
Requirements: pip install vispy pyqt6 numpy
"""

import os
import hashlib
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ----------------- CONFIG -----------------
phi = (1 + np.sqrt(5)) / 2.0
golden_angle_deg = 360.0 / (phi**2)
max_points = 12000
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2.0*np.pi
gamma_echo = 0.88   # echo scale toward origin

base_map = {'A':0,'T':1,'G':2,'C':3}
geometries = [
    (0,'red',1),(1,'green',2),(2,'blue',3),(3,'violet',4),
    (4,'orange',5),(5,'indigo',6),(6,'purple',7),(7,'white',8)
]

# ----------------- VESSEL CONFIG -----------------
vessel_center = np.array([0.0, 0.0, 4.0])
vessel_radius = 25.0
pressure_k = 0.08

def apply_vessel_pressure(points):
    """Soft spherical vessel pressure to Nx3 points"""
    vecs = points - vessel_center
    dists = np.linalg.norm(vecs, axis=1)
    over = dists > vessel_radius
    forces = np.zeros_like(points)
    if np.any(over):
        forces[over] = -pressure_k * (dists[over]-vessel_radius)[:,None] * (vecs[over]/dists[over][:,None])
    return points + forces

# ----------------- LOAD GENOME -----------------
fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError("FASTA not found: " + fasta_path)

def load_genome(path):
    seq = []
    with open(path,'r') as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ----------------- DETERMINISTIC TRAVERSAL -----------------
def build_traversal(seq):
    k = 5
    keys = []
    for i in range(len(seq)):
        kmer = ''.join(seq[(i+j)%len(seq)] for j in range(k))
        h = hashlib.sha256(kmer.encode('ascii')).digest()
        key = int.from_bytes(h[:8],'little')
        keys.append(key)
    return np.argsort(np.array(keys,dtype=np.uint64)).tolist()

traversal = build_traversal(genome_seq)

# ----------------- VISPY SETUP -----------------
canvas = scene.SceneCanvas(keys='interactive', size=(1200,800), bgcolor='black', show=False)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

accum_s1 = np.zeros((max_points,3),dtype=np.float64)
accum_s2 = np.zeros((max_points,3),dtype=np.float64)
write_ptr = 0
filled = 0

empty_pos = np.zeros((max_points,3),dtype=np.float32)
strand1_vis = Line(pos=empty_pos, color=(1,1,1,0.7), width=2, parent=view.scene)
strand2_vis = Line(pos=empty_pos, color=(1,1,1,0.7), width=2, parent=view.scene)

organelles = []
labels = []
centers = []

# ---------- LATTICE LINKS ----------
link_lines = Line(pos=np.zeros((2,3),dtype=np.float32),
                  color=(1,1,0,0.4), width=1, connect='segments', parent=view.scene)
recursive_links = []  # store Line objects for φ-echoes

progress_text = Text("0%", pos=[0,0,20], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

# ----------------- FASTA-DRIVEN HELPERS -----------------
def wrap_idx(i): return i % genome_len

def seq_triplet_values(idx):
    return [base_map[genome_seq[wrap_idx(idx+j)]] for j in range(3)]

def genome_noise(idx,dim):
    tri = seq_triplet_values(idx)
    v = np.array(tri,dtype=np.float64)
    if v.sum()==0: nv=v
    else: nv=v/(v.max()+1.0)
    kmer = ''.join(genome_seq[wrap_idx(idx+j)] for j in range(7))
    h = hashlib.sha256(kmer.encode('ascii')).digest()
    mag = (int.from_bytes(h[:4],'little')%1000)/1000.0
    scale = 0.002*(dim+1)*(0.2+0.8*mag)
    return nv*scale

def organelle_tension(idx):
    tri_vals = seq_triplet_values(idx)
    s = sum(tri_vals)
    kmer = ''.join(genome_seq[wrap_idx(idx+j)] for j in range(5))
    h = hashlib.sha256(kmer.encode('ascii')).digest()
    sign = 1.0 if (h[0]%2==0) else -1.0
    base_t = (s%7)/7.0
    return sign*(0.03*np.sin(base_t*phi*2.0))

def geometry_angle_for_base(base):
    return {
        'A': golden_angle_deg,
        'T': golden_angle_deg*1.3,
        'G': golden_angle_deg*0.8,
        'C': golden_angle_deg*1.6
    }.get(base,golden_angle_deg)

def radius_from_base(idx,base):
    kmer = ''.join(genome_seq[wrap_idx(idx+j)] for j in range(6))
    h = hashlib.sha256(kmer.encode('ascii')).digest()
    val = int.from_bytes(h[:4],'little')%1000
    frac = val/999.0
    base_num = base_map.get(base,0)
    radius = core_radius*(0.55 + 0.45*(1.0 - (frac*(base_num+1)/4.0)))
    return max(0.5,radius)

def genome_phase(idx):
    W=16
    window = [genome_seq[wrap_idx(idx+i)] for i in range(W)]
    freqs = np.array([window.count(b) for b in base_map.keys()],dtype=float)
    s = freqs.sum()
    freq = freqs/s if s!=0 else freqs
    entropy = -np.sum(freq*np.log2(freq+1e-9))
    kmer = ''.join(window[:5])
    h = hashlib.sha256(kmer.encode('ascii')).digest()
    add = (h[0]%60)-30
    return (entropy*40.0*phi + add)%360.0

def spawn_organelle(idx,center):
    val = ord(genome_seq[wrap_idx(idx)]) % len(geometries)
    geom_dim,color,verts = geometries[val]
    noise = genome_noise(idx,geom_dim)*500.0
    pos = (center+noise).astype(np.float64)
    pos = apply_vessel_pressure(pos.reshape(1,3)).reshape(3,)
    rgba = list(Color(color).rgba)
    rgba[3]=1.0
    mark = Markers(pos=pos.reshape(1,3).astype(np.float32),
                   face_color=rgba, edge_color=None, size=6, parent=view.scene)
    return {"marker":mark,"positions":pos.reshape(1,3),"color":rgba,"seed_idx":idx}

# ----------------- UPDATE LOOP -----------------
tick = 0

def update(ev):
    global tick, write_ptr, filled, accum_s1, accum_s2, organelles, centers, recursive_links

    trav_idx = traversal[tick%genome_len]
    base = genome_seq[wrap_idx(trav_idx)]
    dim = base_map.get(base,0)

    theta = (trav_idx * np.radians(geometry_angle_for_base(base)))
    a1 = np.radians(geometry_angle_for_base(base))
    a2 = np.radians(-geometry_angle_for_base(base)*0.9)
    r = radius_from_base(trav_idx,base)
    n1 = genome_noise(trav_idx,dim)
    n2 = genome_noise(trav_idx+3,dim)
    z = np.sin((trav_idx/float(genome_len))*np.pi*4.0)*2.0 + (trav_idx/float(genome_len))*8.0

    p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                   r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                   z],dtype=np.float64)+n1
    p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                   r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                   z],dtype=np.float64)+n2

    # Vessel confinement
    p1 = apply_vessel_pressure(p1.reshape(1,3)).reshape(3,)
    p2 = apply_vessel_pressure(p2.reshape(1,3)).reshape(3,)

    accum_s1[write_ptr,:] = np.round(p1,7)
    accum_s2[write_ptr,:] = np.round(p2,7)
    write_ptr = (write_ptr+1)%max_points
    filled = min(filled+1,max_points)

    # Visualize strands
    if filled<max_points:
        vis_slice = np.vstack([accum_s1[:filled], np.full((max_points-filled,3),np.nan,dtype=np.float64)])
        vis_slice2 = np.vstack([accum_s2[:filled], np.full((max_points-filled,3),np.nan,dtype=np.float64)])
    else:
        idxs = np.concatenate([np.arange(write_ptr,max_points), np.arange(0,write_ptr)])
        vis_slice = accum_s1[idxs]
        vis_slice2 = accum_s2[idxs]

    strand1_vis.set_data(np.nan_to_num(vis_slice.astype(np.float32),nan=0.0))
    strand2_vis.set_data(np.nan_to_num(vis_slice2.astype(np.float32),nan=0.0))

    # Spawn organelles occasionally
    kmer = ''.join(genome_seq[wrap_idx(trav_idx+i)] for i in range(9))
    h = hashlib.sha256(kmer.encode('ascii')).digest()
    if (h[0]%50)==0:
        center = (p1+p2)/2.0
        organelles.append(spawn_organelle(trav_idx,center))
        centers.append(center)

    # Update organelles
    if filled>0:
        lattice_snapshot = vis_slice if filled>=max_points else vis_slice[:filled]
        if lattice_snapshot.size!=0:
            lattice_index = trav_idx%lattice_snapshot.shape[0]
            lattice_nodes = lattice_snapshot
            new_organelles = []
            for org in organelles:
                seed = org['seed_idx']
                tension = organelle_tension(seed)
                new_positions=[]
                for p in org['positions']:
                    nearest = lattice_nodes[lattice_index]
                    dir_vec = (nearest-p)
                    rest_offset = genome_noise(seed+11,1)*200.0
                    newp = p + dir_vec*tension + rest_offset*0.001
                    newp = apply_vessel_pressure(newp.reshape(1,3)).reshape(3,)
                    new_positions.append(np.round(newp,7))
                org['positions'] = np.array(new_positions,dtype=np.float64)
                org['marker'].set_data(pos=org['positions'].astype(np.float32),
                                       face_color=org['color'], size=6)
                new_organelles.append(org)
            organelles=new_organelles

    # ----------------- φ-ECHO LATTICE -----------------
    for i,c1 in enumerate(centers):
        for j in range(i+1,len(centers)):
            c2 = centers[j]
            echo1 = vessel_center + gamma_echo*(c1-vessel_center)
            echo2 = vessel_center + gamma_echo*(c2-vessel_center)
            pts = np.vstack([echo1,echo2])
            link = Line(pos=pts.astype(np.float32), color=(1,1,0,0.1),
                        width=1, connect='segments', parent=view.scene)
            recursive_links.append(link)

    # Labels
    if (ord(base)%37)==0:
        cen=(p1+p2)/2.0
        geom_dim,color,verts=geometries[dim%len(geometries)]
        lbl=Text(f"{base}:{geom_dim}", pos=cen+np.array([0,0,0.3]),
                 color=color,font_size=10,bold=True,anchor_x='center',parent=view.scene)
        labels.append(lbl)

    # Camera
    phase=genome_phase(trav_idx)
    view.camera.azimuth=phase
    view.camera.elevation=15.0 + 8.0*np.sin(phase/57.2957795)

    # Progress
    progress = (tick%genome_len)/float(genome_len)*100.0
    progress_text.text = f"{progress:.4f}%"

    tick+=1

# ----------------- START -----------------
timer = app.Timer(interval=0.016, connect=update, start=True)
if __name__=="__main__":
    canvas.show()
    app.run()
