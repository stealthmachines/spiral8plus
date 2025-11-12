# dna_holo_cell_gpu_fixed.py
# GPU-accelerated φ-spiral, genome-driven Human Genome-like cell (instanced rendering)
# Install: pip install vispy pyqt6 numpy
# Run: python dna_holo_cell_gpu_fixed.py

import os
import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

# ---------- CONFIG ----------
phi = (1 + np.sqrt(5)) / 2.0
golden_angle_deg = 360.0 / (phi ** 2)  # ~137.507
core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 16000          # number of instanced points per strand
MAX_TEX = 65536             # genome texture length (windowed)
POINT_SIZE = 4.0

# nucleotide → numeric mapping (0..3)
base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

angles = np.array([i * golden_angle_deg for i in range(8)], dtype=np.float32)

# ---------- LOAD GENOME (map to uint8 array) ----------

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
    """Automatically find the Human Genome FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\GCA_000001405.29\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file in ncbi_dataset directory")


def load_genome_codes(fasta_file, max_len=None):
    seq_codes = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            for ch in line.strip().upper():
                if ch in base_map:
                    seq_codes.append(base_map[ch])
                else:
                    seq_codes.append(3)  # treat unknown as 'C' (safe fallback)
                if max_len and len(seq_codes) >= max_len:
                    return np.array(seq_codes, dtype=np.uint8)
    return np.array(seq_codes, dtype=np.uint8)

fasta_path = find_human_fasta()
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: " + fasta_path)

# We'll load up to MAX_TEX bases for the GPU texture window.
genome_codes_full = load_genome_codes(fasta_path, max_len=None)
genome_len_full = len(genome_codes_full)
print("Full genome length (codes):", genome_len_full)

# Use a sliding window texture length (MAX_TEX or available)
tex_len = min(MAX_TEX, genome_len_full)
genome_codes = genome_codes_full[:tex_len].astype(np.uint8)
print("Using genome texture length:", tex_len)

# ---------- GLSL SHADERS ----------
vertex_shader = """\
/* Instanced vertex shader: compute position for each instance from genome texture */
uniform float u_frame;
uniform float u_core_radius;
uniform float u_strand_sep;
uniform float u_genome_start;
uniform float u_genome_len;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_angles[8];
uniform float u_timewarp;

attribute float a_id;    // instance id 0..(N-1)
attribute float a_which; // 0 = strand1, 1 = strand2, 2 = donut center, 3 = organelle

uniform sampler2D u_genome_tex; // 1D genome data as a 2D texture with height 1

varying float v_base_code;
varying float v_alpha;

float my_mod(float a, float b) {
    return a - b * floor(a / b);
}

void main() {
    float idx = a_id + u_frame; // advancing read head by frame
    float wrapped = my_mod(u_genome_start + idx, u_genome_len); // 0..u_genome_len-1
    float texcoord = (wrapped + 0.5) / u_genome_len; // normalized [0..1]

    vec2 uv = vec2(texcoord, 0.5);
    float v = texture2D(u_genome_tex, uv).r; // normalized 0..1
    float base_code = floor(v * 255.0 + 0.5);
    v_base_code = base_code;

    int dim = int(mod(base_code, 8.0));

    float core_r = u_core_radius * (1.0 - pow((wrapped / u_genome_len), 1.4));
    core_r = max(core_r, 0.5);

    float theta = radians(%f) * wrapped;

    float a1 = radians(u_angles[dim]);
    float a2 = radians(-u_angles[dim]);

    vec3 p1 = vec3(
        core_r * cos(theta) * cos(a1) - core_r * sin(theta) * sin(a1),
        core_r * sin(theta) * cos(a1) + core_r * cos(theta) * sin(a1),
        sin(wrapped / u_genome_len * 3.14159 * 4.0) * 1.6 + (wrapped / u_genome_len) * 6.0
    );

    vec3 p2 = vec3(
        core_r * cos(theta) * cos(a2) - core_r * sin(theta) * sin(a2) + u_strand_sep,
        core_r * sin(theta) * cos(a2) + core_r * cos(theta) * sin(a2) - u_strand_sep,
        sin(wrapped / u_genome_len * 3.14159 * 4.0) * 1.6 + (wrapped / u_genome_len) * 6.0
    );

    float t = u_timewarp * 0.001 * u_frame;
    p1.x += 0.02 * cos(t);
    p2.x -= 0.02 * cos(t);
    p1 += vec3(0.0, 0.0, sin(u_frame * 0.003) * 0.5);
    p2 += vec3(0.0, 0.0, -sin(u_frame * 0.003) * 0.3);

    vec3 pos = vec3(0.0);
    if (int(a_which + 0.5) == 0) {
        pos = p1;
        v_alpha = 0.9;
    } else if (int(a_which + 0.5) == 1) {
        pos = p2;
        v_alpha = 0.9;
    } else if (int(a_which + 0.5) == 2) {
        pos = (p1 + p2) * 0.5;
        v_alpha = 0.5;
    } else {
        float idf = a_id;
        float rx = fract(sin(idf * 12.9898) * 43758.5453) - 0.5;
        float ry = fract(sin(idf * 78.233) * 43758.5453) - 0.5;
        float rz = fract(sin(idf * 45.164) * 43758.5453) - 0.5;
        pos = (p1 + p2) * 0.5 + vec3(rx, ry, rz) * (0.8 + float(dim) * 0.05);
        v_alpha = 1.0;
    }

    vec4 world = u_model * vec4(pos, 1.0);
    vec4 viewp = u_view * world;
    gl_Position = u_proj * viewp;

    gl_PointSize = %f;
}
""" % (golden_angle_deg, POINT_SIZE)

fragment_shader = """\
varying float v_base_code;
varying float v_alpha;

void main() {
    float b = v_base_code;
    vec3 col = vec3(0.2, 0.8, 0.8);
    if (mod(b, 8.0) < 1.0) col = vec3(1.0, 0.2, 0.2);
    else if (mod(b, 8.0) < 2.0) col = vec3(0.2, 1.0, 0.2);
    else if (mod(b, 8.0) < 3.0) col = vec3(0.6, 0.2, 1.0);
    else if (mod(b, 8.0) < 4.0) col = vec3(0.6, 0.6, 1.0);
    else if (mod(b, 8.0) < 5.0) col = vec3(0.9, 0.6, 0.2);
    else if (mod(b, 8.0) < 6.0) col = vec3(0.5, 0.1, 0.7);
    else col = vec3(1.0, 1.0, 1.0);

    gl_FragColor = vec4(col * (0.6 + 0.4 * fract(v_base_code)), v_alpha);
}
"""

# ---------- CANVAS ----------
class HoloCanvas(app.Canvas):
    def __init__(self):
        # Initialize canvas first (resize may be called early)
        app.Canvas.__init__(self, keys='interactive', size=(1200, 800), title='Holographic DNA Cell (GPU)')

        # Create program immediately
        self.program = gloo.Program(vertex_shader, fragment_shader)

        # safe defaults
        self._frame = 0.0
        self._genome_start = 0.0
        self._genome_len = float(tex_len)
        self._timewarp = 1.0

        # create an array of instance ids 0..N-1 (float attribute)
        N = MAX_POINTS
        ids = np.arange(N, dtype=np.float32)
        which = np.zeros_like(ids, dtype=np.float32)  # we'll set per draw call
        self.ids_buf = gloo.VertexBuffer(ids)
        self.which_buf = gloo.VertexBuffer(which)

        # set common uniforms (ensure correct shapes)
        self.program['u_core_radius'] = float(core_radius)
        self.program['u_strand_sep'] = float(strand_sep)
        self.program['u_genome_start'] = float(self._genome_start)
        self.program['u_genome_len'] = float(self._genome_len)
        self.program['u_timewarp'] = float(self._timewarp)

        # IMPORTANT: u_angles must be shape (8,1) to satisfy gloo on some backends
        self.program['u_angles'] = angles.astype(np.float32).reshape(8, 1)

        # model/view/proj (will update on resize)
        self.model = np.eye(4, dtype=np.float32)
        self.view = translate((0, 0, -40)).astype(np.float32)
        self.proj = perspective(45.0, self.size[0] / float(self.size[1]), 1.0, 1000.0).astype(np.float32)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_proj'] = self.proj

        # genome texture: encode codes 0..3 into red channel 0..255
        tex_width = tex_len
        tex_data = np.zeros((1, tex_width, 4), dtype=np.uint8)
        # store 0..255 values in red channel so texture2D(...).r yields 0..1 where r*255 == code
        tex_data[0, :tex_len, 0] = genome_codes
        self.tex = gloo.Texture2D(tex_data, interpolation='nearest', internalformat='rgba8')
        self.program['u_genome_tex'] = self.tex

        # bind attributes
        self.program['a_id'] = self.ids_buf
        self.program['a_which'] = self.which_buf

        # Now connect resize safely (program exists)
        self.events.resize.connect(self.on_resize)

        gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'),
                       depth_test=True, polygon_offset_fill=False)

        self._timer = app.Timer(0.016, connect=self.on_timer, start=True)
        self.show()

    def on_resize(self, event):
        # Defensive: if program not ready for any reason, skip
        if not hasattr(self, 'program') or self.program is None:
            return
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.proj = perspective(45.0, width / float(height), 1.0, 1000.0).astype(np.float32)
        self.program['u_proj'] = self.proj

    def on_timer(self, event):
        # update frame and uniforms
        self._frame += 1.0
        if hasattr(self, 'program') and self.program is not None:
            self.program['u_frame'] = float(self._frame)
            self.program['u_genome_start'] = float(self._genome_start)
            self.program['u_timewarp'] = float(self._timewarp)

            # camera rotation
            az = (self._frame * 0.12) % 360.0
            el = 15.0 + 8.0 * np.sin(self._frame * 0.002)
            m = translate((0, 0, -40)).astype(np.float32)
            m = rotate(az, (0, 1, 0)).astype(np.float32) @ m
            m = rotate(el, (1, 0, 0)).astype(np.float32) @ m
            self.program['u_view'] = m

        self.update()

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        N = MAX_POINTS

        # draw strand1 (which = 0)
        which_data = np.zeros(N, dtype=np.float32)
        self.which_buf.set_data(which_data)
        self.program['a_which'] = self.which_buf
        self.program.draw('points', count=N)

        # draw strand2 (which = 1)
        which_data[:] = 1.0
        self.which_buf.set_data(which_data)
        self.program['a_which'] = self.which_buf
        self.program.draw('points', count=N)

        # draw donut centers (which = 2)
        which_data[:] = 2.0
        self.which_buf.set_data(which_data)
        self.program['a_which'] = self.which_buf
        self.program.draw('points', count=N)

        # draw organelles (which = 3)
        organ_n = max(128, N // 8)
        which_data2 = np.full(N, 3.0, dtype=np.float32)
        which_data2[organ_n:] = 999.0
        self.which_buf.set_data(which_data2)
        self.program['a_which'] = self.which_buf
        self.program.draw('points', count=organ_n)

    def on_key_press(self, event):
        if event.key.name == 'Space':
            self._genome_start = (self._genome_start + 256.0) % float(self._genome_len)
            self.program['u_genome_start'] = float(self._genome_start)
            print("Genome start ->", int(self._genome_start))
        elif event.key.name == 'Up':
            self._timewarp *= 1.1
            print("timewarp", self._timewarp)
        elif event.key.name == 'Down':
            self._timewarp /= 1.1
            print("timewarp", self._timewarp)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Human Genome GRCh38.p14 Visualization")
    print("="*60)
    if 'genome_len' in dir():
        print(f"Genome length: {genome_len:,} nucleotides")
    print("="*60)

    c = HoloCanvas()
    app.run()
