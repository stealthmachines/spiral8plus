# ecoli_resonance.py
# --------------------------------------------------------------
# Composite φ-harmonic spiral + real-time resonance lattice overlay
# Computes variance across all 24 A/T/G/C → geometry mappings
# and paints low-variance (resonant) zones teal/orange and
# high-variance zones purple. Uses Vispy for OpenGL rendering.
# Install: pip install vispy pyqt6 numpy
# --------------------------------------------------------------

import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- PARAMETERS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

bases = ['A', 'T', 'G', 'C']
all_mappings = [dict(zip(bases, p)) for p in itertools.permutations([1,2,3,4])]
assert len(all_mappings) == 24

# geometry palette (kept for rung/color reference)
geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'green',        'Line',         0.008262, 2),
    (3, 'E', 'violet',       'Triangle',     0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'white',        'Octacube',     0.012345, 16),
]

angles = [i * golden_angle_deg for i in range(8)]
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2*np.pi
N = 600  # points per strand/segment

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome("ecoli_k12.fasta")
genome_len = len(genome_seq)

# ---------- VISPY: composite canvas ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1024,768),
                                bgcolor='black', title="Composite Spiral + Resonance Lattice")
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

# Composite lines
comp_line1 = Line(pos=np.zeros((N,3)), color=(0,1,1,0.7), width=2, parent=comp_view.scene)
comp_line2 = Line(pos=np.zeros((N,3)), color=(1,0.5,0,0.7), width=2, parent=comp_view.scene)

# Resonance markers (positions & colors updated each frame)
res_markers = Markers(pos=np.zeros((N,3)), face_color=np.zeros((N,4)), edge_color=None,
                      size=6, parent=comp_view.scene)

# Rungs / echoes collections (composite-based)
collections = {'rungs': [], 'echoes': [], 'links': [], 'labels': [], 'centers': [], 'emerged': []}

# Overlay text
legend_text = Text("Resonance: teal = low variance (agreement) → orange → purple = high variance",
                   pos=[-10, -9, 0], color='white', font_size=10, parent=comp_view.scene)
pct_text = Text("", pos=[-10, -10, 0], color='white', font_size=12, bold=True, parent=comp_view.scene)

frame = 0

# ---------- helper: variance -> RGBA color mapping ----------
def variance_to_rgba(var_arr):
    """
    var_arr: 1D array of non-negative variance values (shape (N,))
    returns RGBA array shape (N,4)
    color mapping:
      low (<= q25) -> teal (~0,0.9,0.8)
      mid -> blend teal -> orange
      high (>= q75) -> purple (~0.7,0.25,0.9)
    """
    # Normalize to 0..1 by robust scaling
    v_min = np.percentile(var_arr, 1)
    v_max = np.percentile(var_arr, 99)
    v = (var_arr - v_min) / max(1e-9, (v_max - v_min))
    v = np.clip(v, 0.0, 1.0)

    # color anchors
    teal = np.array([0.0, 0.9, 0.8])
    orange = np.array([1.0, 0.55, 0.0])
    purple = np.array([0.7, 0.25, 0.9])

    rgba = np.zeros((len(v), 4), dtype=np.float32)
    # low -> teal (v near 0)
    # mid -> orange (v around 0.5)
    # high -> purple (v near 1)
    # We'll do a two-segment blend: [0..0.5] teal->orange, [0.5..1] orange->purple
    low_mask = v <= 0.5
    high_mask = v > 0.5

    # blend 0..0.5
    if low_mask.any():
        t = v[low_mask] / 0.5
        colors = (1-t)[:,None]*teal + t[:,None]*orange
        rgba[low_mask,:3] = colors
        rgba[low_mask,3] = 0.9 - 0.4 * t  # more solid at teal end, slightly more transparent toward orange

    # blend 0.5..1
    if high_mask.any():
        t = (v[high_mask] - 0.5) / 0.5
        colors = (1-t)[:,None]*orange + t[:,None]*purple
        rgba[high_mask,:3] = colors
        rgba[high_mask,3] = 0.6 + 0.3 * t  # slightly more solid toward purple

    return rgba

# ---------- MAIN UPDATE ----------
def update(ev):
    global frame, collections
    frame += 1
    t = np.linspace(frame, frame + N - 1, N)  # N sample points along the strand

    # accumulate positions for each mapping: shape (M, N, 3)
    M = len(all_mappings)
    pos1_all = np.zeros((M, N, 3), dtype=np.float32)
    pos2_all = np.zeros((M, N, 3), dtype=np.float32)

    for mi, base_map in enumerate(all_mappings):
        s1 = np.zeros((N,3), dtype=np.float32)
        s2 = np.zeros((N,3), dtype=np.float32)
        for ii, tt in enumerate(t):
            idx = int(tt) % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base, 1) - 1
            _, _, _, _, alpha, _ = geometries[dim]

            r = core_radius * (1 - (tt/genome_len)**1.5)
            r = max(r, 0.5)
            theta = tt * np.radians(golden_angle_deg)
            z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len) * 8

            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])

            x1 = r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1)
            y1 = r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1)
            x2 = r*np.cos(theta)*np.cos(a2) - r*np.sin(theta)*np.sin(a2) + strand_sep
            y2 = r*np.sin(theta)*np.cos(a2) + r*np.cos(theta)*np.sin(a2) - strand_sep

            s1[ii] = [x1, y1, z]
            s2[ii] = [x2, y2, z]

        pos1_all[mi] = s1
        pos2_all[mi] = s2

    # composite average
    comp_s1 = pos1_all.mean(axis=0)
    comp_s2 = pos2_all.mean(axis=0)

    # update composite lines
    comp_line1.set_data(comp_s1)
    comp_line2.set_data(comp_s2)

    # compute per-point positional variance across mappings (use radial variance or 3D variance)
    # Here we use 3D std norm at each sample point across mappings
    diffs1 = pos1_all - comp_s1[None, :, :]  # shape (M,N,3)
    diffs2 = pos2_all - comp_s2[None, :, :]
    var1 = np.sqrt(np.sum(diffs1**2, axis=2)).std(axis=0)  # std of distances across M at each point
    var2 = np.sqrt(np.sum(diffs2**2, axis=2)).std(axis=0)

    # combine the two strands' variance by simple average for overlay
    var_combined = 0.5 * (var1 + var2)

    # convert variance -> RGBA
    colors = variance_to_rgba(var_combined)

    # update resonance markers at composite positions (use comp_s1 for coordinates)
    res_markers.set_data(pos=comp_s1, face_color=colors, size=6)

    # ---------- EMERGE RUNGS on composite (consensus dim) ----------
    # compute consensus dimension for current frame index across mappings
    cur_idx = frame % genome_len
    dims = np.array([m.get(genome_seq[cur_idx],1)-1 for m in all_mappings])
    cur_dim = int(np.round(dims.mean()))
    if cur_dim < 0: cur_dim = 0
    if cur_dim >= len(geometries): cur_dim = len(geometries)-1

    if frame % 20 == 0 and cur_dim > len(collections['rungs'])-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        collections['emerged'].append(cur_dim)

        start = int((dim/8.0)*N)
        step = max(1, N // (8*verts))
        idxs = np.arange(start, start + verts*step, step)[:verts]
        idxs = idxs[idxs < N]

        pts1 = comp_s1[idxs]
        pts2 = comp_s2[idxs]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.95]

        segs = []
        if verts <= 4:
            for pts in (pts1, pts2):
                for i in range(verts):
                    for j in range(i+1, verts):
                        segs += [pts[i], pts[j]]
            for i in range(verts):
                segs += [pts1[i], pts2[i]]
        else:
            g = int(np.ceil(np.sqrt(verts)))
            for i in range(g):
                for j in range(g):
                    n = min(i*g + j, verts-1)
                    if j+1 < g and n+1 < verts:
                        segs += [pts1[n], pts1[n+1], pts2[n], pts2[n+1]]
                    if i+1 < g and n+g < verts:
                        segs += [pts1[n], pts1[n+g], pts2[n], pts2[n+g]]
            for n in range(verts):
                segs += [pts1[n], pts2[n]]

        if segs:
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', parent=comp_view.scene)
            collections['rungs'].append(line)

        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=8, parent=comp_view.scene)
        collections['rungs'].append(mark)

        cen = all_pts.mean(axis=0)
        collections['centers'].append(cen)
        lbl = Text(f"{genome_seq[cur_idx]}: {name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True,
                   anchor_x='center', parent=comp_view.scene)
        collections['labels'].append(lbl)

        # echo
        if len(collections['emerged']) > 1:
            prev = collections['emerged'][-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(pos=echo_pts, face_color=prev_col, size=5, parent=comp_view.scene)
            collections['echoes'].append(echo)

        # inter-rung links
        if len(collections['centers']) > 1:
            prev_c = collections['centers'][-2]
            segs = []
            for i in range(min(6, len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=comp_view.scene)
            collections['links'].append(link)

    # ---------- CAMERA + OVERLAY ----------
    comp_view.camera.azimuth = frame * 0.18
    comp_view.camera.elevation = 20 + 5 * np.sin(frame * 0.005)
    pct = 100.0 * frame / genome_len
    pct_text.text = f"Frame: {frame}  |  Genome complete: {pct:.2f}%"
    comp_canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
