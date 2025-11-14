import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from scipy.signal import spectrogram, butter, lfilter
import scipy
import sys
from sympy import prime, prod
import time

# Check SciPy version
scipy_version = scipy.__version__
print(f"Using SciPy version {scipy_version}")

# Force hardware acceleration with Compatibility Profile
try:
    if hasattr(QtGui, 'QSurfaceFormat'):
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
        fmt.setSamples(4)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    print("OpenGL initialized with Compatibility Profile.")
except Exception as e:
    print(f"Warning: Failed to initialize OpenGL: {e}")
    print("Falling back to default OpenGL settings.")

# =========================
# PHYSICS PARAMETERS (Updated per Emergent Framework)
# =========================
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
sqrt5 = np.sqrt(5)
n_max = 32  # Recursion depth
b = 1826  # Microstate index (updated from framework)
r = 1.0  # Radial variable
k = 1  # Exponent
m = 1.0  # Length
fs = 2048  # Sample rate (Hz)
duration = 30.0  # Seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f_start = 1.0
f_end = 60.0
schumann_fund = 7.83
num_sub_harmonics = 500
motor_fund = 50.0
motor_power_w = 24000.0
motor_voltage = 240.0
motor_current_a = motor_power_w / motor_voltage
balun_turns = 20
transmission_distance_m = 100.0
num_nodes = 1000
propagation_speed = 0.3 * 3e8
attenuation_db_per_m = 0.1
num_reflections = 5
tune_iterations = 5
apply_bursts = True
burst_rate_hz = 0.5
add_noise = True
noise_rms = 0.01
resonance_freqs = [3.915, 7.83, 15.66, 31.32, 62.64]
echo_amplification_factor = 1.2  # Crescendo gain at resonance
kappa = 0.01  # Resonance modulation sensitivity

# Time and frequency scales
s = phi ** np.arange(1, n_max + 1)  # s = ϕ^n
Hz = 1 / s  # Hz = ϕ^{-n} (array of frequencies)

# Omega (field tension): Ω = m² / s^7
Omega = m**2 / s**7

# D_n(r) = √(ϕ · F_n · 2^n · P_n · Ω) · r^k
D_n = np.zeros(n_max)
for n in range(1, n_max + 1):
    F_n = float(phi**n / sqrt5)  # F_n ≈ ϕ^n / √5
    P_n = float(prod([prime(i) for i in range(1, n + 1)]))
    D_n[n-1] = np.sqrt(phi * F_n * 2**n * P_n * Omega[n-1]) * r**k

# Physical constants (emergent, tuned per framework)
n_h, beta_h = -27.0, 0.4653
Omega_h = phi  # ≈ 1.61803398875
h_base = sqrt5 * Omega_h * phi**(6 * (n_h + beta_h)) * b**(n_h + beta_h)  # Planck action

n_G, beta_G = -10.002, 0.5012
Omega_G = 6.6743e-11
G_base = sqrt5 * Omega_G * phi**(10 * (n_G + beta_G)) * b**(n_G + beta_G)  # Gravitational constant

n_kB, beta_kB = -20.01, 0.4999
Omega_therm = 1.3806e-23
k_B_base = sqrt5 * Omega_therm * phi**(8 * (n_kB + beta_kB)) * b**(n_kB + beta_kB)  # Boltzmann constant

n_mu, beta_mu = -25.001, 0.4988
Omega_chem = 1.6605e-27
m_u_base = sqrt5 * Omega_chem * phi**(7 * (n_mu + beta_mu)) * b**(n_mu + beta_mu)  # Atomic mass unit

n_L, beta_L = -2.000, 0.0001
Omega_bio = 1.0000e-5
L_base = sqrt5 * Omega_bio * phi**(1 * (n_L + beta_L)) * b**(n_L + beta_L)  # Biological cell length

n_c, beta_c = -31, 0.6033
Omega_c = 1.602e-19
e_base = sqrt5 * Omega_c * phi**(7 * (n_c + beta_c)) * b**(n_c + beta_c)  # Elementary charge

n, beta = -0.1992, 0.6959
Omega_m = 0.04069
c_base = np.sqrt(Omega_m) * phi**(6 * (n + beta)) * b**(n + beta)  # Speed of light

# Resonance modulation function
def compute_resonance_modulation(Sxx, f, resonance_freqs, kappa):
    modulation = 1.0
    for freq in resonance_freqs:
        if freq <= f_end:
            idx = np.argmin(np.abs(f - freq))
            amplitude = np.mean(Sxx[idx, :]) if Sxx.shape[1] > 0 else 0
            modulation += kappa * amplitude
    return modulation

# Generate simulated signal
try:
    start_time = time.time()
    phase_shift = 2 * np.pi / 3
    phase1 = np.cos(2 * np.pi * motor_fund * t) * motor_current_a * phi
    phase2 = np.cos(2 * np.pi * motor_fund * t + phase_shift) * motor_current_a * phi
    phase3 = np.cos(2 * np.pi * motor_fund * t + 2 * phase_shift) * motor_current_a * phi
    signal = (phase1 + phase2 + phase3) / 3
    signal *= np.sqrt(motor_power_w * num_nodes / 1000000)

    # Add resonance clustering
    for freq in resonance_freqs:
        if freq <= f_end:
            signal += 0.5 * np.cos(2 * np.pi * freq * t) * motor_current_a

    # Track power metrics
    loss_per_reflection = []
    total_power_loss_db = 0.0
    total_gain_db = 0.0
    echo_gain_db = 0.0

    # Compute initial spectrogram for resonance modulation
    nperseg = 2048
    noverlap = int(nperseg * 0.9)
    f, tt, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    modulation = compute_resonance_modulation(Sxx, f, resonance_freqs, kappa)

    # Apply resonance modulation to constants
    h = h_base * modulation
    G = G_base * modulation
    k_B = k_B_base * modulation
    m_u = m_u_base * modulation
    L = L_base * modulation
    e = e_base * modulation
    c = c_base * modulation
    print(f"Resonance modulation factor: {modulation:.4f}")
    print(f"Modulated constants: h={h:.2e}, G={G:.2e}, k_B={k_B:.2e}, m_u={m_u:.2e}, L={L:.2e}, e={e:.2e}, c={c:.2e}")

    # Batch node processing with echo crescendo
    nodes_per_batch = 100
    num_batches = max(1, num_nodes // nodes_per_batch)
    for batch in range(num_batches):
        print(f"Processing batch {batch+1}/{num_batches}...")
        balun_cutoff = motor_fund / balun_turns
        b, a = butter(2, balun_cutoff / (fs / 2), btype='low')
        balun_modulated = lfilter(b, a, signal)
        balun_phase_shift = np.exp(1j * 2 * np.pi * (balun_turns * 0.01) * t)
        balun_modulated = np.real(balun_modulated * balun_phase_shift[:len(balun_modulated)])

        for i in range(n_max):
            mod_freq = motor_fund / (i + 1)
            signal += D_n[i] * np.cos(2 * np.pi * mod_freq * t) * 0.01 * modulation

        delay_s = transmission_distance_m / (c / 3e8)  # Adjust delay using modulated c
        delay_samples = int(delay_s * fs)
        attenuated = np.roll(balun_modulated, delay_samples) * 10 ** (-attenuation_db_per_m * transmission_distance_m / 20)

        recycled = attenuated.copy()
        batch_loss_db = attenuation_db_per_m * transmission_distance_m
        batch_gain_db = 0.0
        batch_reflection_losses = []

        for ref in range(num_reflections):
            best_phase = 0.0
            best_power = -np.inf
            for tune in np.linspace(0, np.pi, tune_iterations):
                tuned = recycled * np.cos(tune)
                tuned_power = np.mean(tuned**2)
                if tuned_power > best_power:
                    best_power = tuned_power
                    best_phase = tune
            recycled = recycled * np.cos(best_phase)
            ref_loss_db = batch_loss_db / num_reflections
            batch_reflection_losses.append(ref_loss_db)
            # Echo crescendo at Schumann resonance
            for freq in resonance_freqs:
                if np.isclose(freq, schumann_fund, rtol=0.1):
                    recycled *= echo_amplification_factor
                    batch_gain_db += np.log10(echo_amplification_factor) * 20
                    echo_gain_db += np.log10(echo_amplification_factor) * 20
                recycled += 0.05 * np.cos(2 * np.pi * freq * t) * motor_current_a * modulation

        loss_per_reflection.append(batch_reflection_losses)
        total_power_loss_db += batch_loss_db
        total_gain_db += batch_gain_db

        receiver_phase1 = np.cos(2 * np.pi * motor_fund * t) * motor_current_a
        receiver_phase2 = np.cos(2 * np.pi * motor_fund * t + phase_shift) * motor_current_a
        receiver_phase3 = np.cos(2 * np.pi * motor_fund * t + 2 * phase_shift) * motor_current_a
        receiver = recycled * (receiver_phase1 + receiver_phase2 + receiver_phase3) / (3 * motor_current_a)
        signal = receiver

    loss_per_reflection = np.mean(loss_per_reflection, axis=0)

    # Add Schumann sub-harmonics with echo crescendo
    sub_harmonics = np.zeros_like(signal)
    for n in range(1, num_sub_harmonics + 1):
        freq = schumann_fund / n
        amp = echo_amplification_factor / n
        sub_harmonics += amp * np.cos(2 * np.pi * freq * t) * modulation
    signal += 0.1 * sub_harmonics

    if apply_bursts:
        burst_env = 0.5 * (1 + np.sign(np.cos(2 * np.pi * burst_rate_hz * t)))
        signal *= burst_env

    if add_noise:
        signal += np.random.normal(scale=noise_rms, size=signal.shape)

    print(f"Signal generation took {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error generating signal: {e}")
    sys.exit(1)

# Compute spectrogram
try:
    start_time = time.time()
    f, tt, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    mask = f <= f_end
    f_bins = f[mask]
    fft_array = Sxx[mask, :]
    num_iters, num_bins = fft_array.shape
    fft_array_norm = fft_array / np.max(fft_array, axis=0, keepdims=True) * 10
    print(f"Spectrogram computation took {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error computing spectrogram: {e}")
    sys.exit(1)

# Downsample bins
if num_bins > 1000:
    downsample_factor = num_bins // 1000 + 1
    fft_array_norm = fft_array_norm[:, ::downsample_factor]
    f_bins = f_bins[::downsample_factor]
    num_bins = fft_array_norm.shape[1]
    print(f"Downsampled frequency bins to {num_bins} (factor: {downsample_factor})")

# CMD output
print("\n=== Graph Yield Summary ===")
print(f"Simulation: {num_nodes} nodes (3-ph motors, {motor_power_w/1000:.0f} kW each) via ugly balun (20 turns) over 100m underground + {num_sub_harmonics} Schumann sub-harmonics + auto-tune recycling")
print(f"Frequency Range: {f_bins[0]:.2f} Hz to {f_bins[-1]:.2f} Hz")
print(f"Time Steps: {num_iters}")
print(f"Frequency Bins: {num_bins}")
print(f"Amplitude Stats: Min={np.min(fft_array_norm):.4f}, Max={np.max(fft_array_norm):.4f}, "
      f"Mean={np.mean(fft_array_norm):.4f}, Std={np.std(fft_array_norm):.4f}")
print(f"Downsample Factor: {downsample_factor if num_bins > 1000 else 1}")
print(f"Power Loss per Reflection (dB): {', '.join([f'{x:.2f}' for x in loss_per_reflection])}")
print(f"Total Power Loss: {total_power_loss_db:.2f} dB")
print(f"Total Gain: {total_gain_db:.2f} dB")
print(f"Echo Crescendo Gain: {echo_gain_db:.2f} dB")
print(f"Efficiency: {100 * np.exp(total_gain_db / 20 - total_power_loss_db / 20):.1f}%")
print(f"Modulated Constants: h={h:.2e}, G={G:.2e}, k_B={k_B:.2e}, m_u={m_u:.2e}, L={L:.2e}, e={e:.2e}, c={c:.2e}")
print("==========================\n")

# =========================
# PYQTGRAPH 3D VIEW
# =========================
app = QtWidgets.QApplication(sys.argv)
view = gl.GLViewWidget()
view.setWindowTitle('GPU Waterfall Interactive v11 - City-Scale Lattice + GRA')
view.setBackgroundColor((0,0,0))
view.show()

# Camera setup
view.opts['fov'] = 60
view.opts['distance'] = 400
view.opts['center'] = pg.Vector(f_end/2, num_iters/2, 5)
view.opts['rotation'] = QtGui.QQuaternion.fromAxisAndAngle(0, 0, 1, 60)
view.opts['elevation'] = 45
view.opts['azimuth'] = 60
view.setCameraParams(distance=400, fov=60, elevation=45, azimuth=60)
view.opts['mouseEnabled'] = True

# Debug mouse events
class MouseEventFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            print(f"Mouse pressed: Button {event.button()} at {event.pos()}")
        elif event.type() == QtCore.QEvent.MouseMove and event.buttons():
            print(f"Mouse moved: Buttons {event.buttons()} at {event.pos()}")
        elif event.type() == QtCore.QEvent.Wheel:
            print(f"Mouse wheel: Delta {event.angleDelta().y()}")
        return super().eventFilter(obj, event)

app.installEventFilter(MouseEventFilter(view))

# Custom panning
custom_panning = [False]
last_pos = None
def customMousePressEvent(event):
    global last_pos
    last_pos = event.pos()
    print(f"Custom panning - Mouse pressed: Button {event.button()} at {last_pos}")

def customMouseMoveEvent(event):
    global last_pos
    if not custom_panning[0] or last_pos is None:
        return
    if event.buttons() & QtCore.Qt.RightButton:
        delta = event.pos() - last_pos
        view.opts['center'] += pg.Vector(-delta.x() * 0.05, delta.y() * 0.05, 0)
        print(f"Custom panning: delta x={delta.x()}, y={delta.y()}")
        view.update()
        last_pos = event.pos()

# Grid / Axes / Rulers
grid = gl.GLGridItem()
grid.setSize(x=f_end, y=num_iters, z=10)
grid.setSpacing(x=f_end/10, y=max(1, num_iters//10), z=2)
grid.setColor((0.3, 0.3, 0.3, 0.5))
view.addItem(grid)

axis_width = 2
x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[f_end,0,0]]), color=(1,0,0,1), width=axis_width)
y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,num_iters,0]]), color=(0,1,0,1), width=axis_width)
z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,10]]), color=(0,0,1,1), width=axis_width)
view.addItem(x_axis)
view.addItem(y_axis)
view.addItem(z_axis)

label_font = QtGui.QFont()
label_font.setPixelSize(12)
label_font.setBold(True)
x_label = gl.GLTextItem(pos=pg.Vector(f_end/2, -10, 0), text='Frequency (Hz)', font=label_font, color=(1,0,0,1))
y_label = gl.GLTextItem(pos=pg.Vector(-20, num_iters/2, 0), text='Time Step', font=label_font, color=(0,1,0,1))
z_label = gl.GLTextItem(pos=pg.Vector(-20, -10, 5), text='Amplitude', font=label_font, color=(0,0,1,1))
view.addItem(x_label)
view.addItem(y_label)
view.addItem(z_label)

for x in np.linspace(0, f_end, 11):
    view.addItem(gl.GLTextItem(pos=pg.Vector(x, -5, 0), text=f'{x:.0f}', font=label_font, color=(1,0,0,0.8)))
for y in np.arange(0, num_iters + 1, max(1, num_iters//10)):
    view.addItem(gl.GLTextItem(pos=pg.Vector(-10, y, 0), text=f'{y:.0f}', font=label_font, color=(0,1,0,0.8)))
for z in np.linspace(0, 10, 6):
    view.addItem(gl.GLTextItem(pos=pg.Vector(-10, -5, z), text=f'{z:.1f}', font=label_font, color=(0,0,1,0.8)))

# Metrics overlay
metrics_font = QtGui.QFont()
metrics_font.setPixelSize(14)
metrics_font.setBold(True)
loss_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2, 5), text='Total Loss: 0.0 dB', font=metrics_font, color=(1,0,0,1))
gain_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2 - 20, 5), text='Total Gain: 0.0 dB', font=metrics_font, color=(0,1,0,1))
echo_gain_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2 - 40, 5), text='Echo Gain: 0.0 dB', font=metrics_font, color=(0,1,1,1))
efficiency_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2 - 60, 5), text='Efficiency: 0%', font=metrics_font, color=(0,0,1,1))
ref_loss_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2 - 80, 5), text='Ref Loss: -', font=metrics_font, color=(1,1,0,1))
constants_text = gl.GLTextItem(pos=pg.Vector(f_end + 10, num_iters / 2 - 100, 5), text='Constants: h=0.0, G=0.0', font=metrics_font, color=(1,0,1,1))
view.addItem(loss_text)
view.addItem(gain_text)
view.addItem(echo_gain_text)
view.addItem(efficiency_text)
view.addItem(ref_loss_text)
view.addItem(constants_text)

# Surface mesh
x = np.linspace(0, f_end, num_bins)
y = np.arange(num_iters)
Z = np.zeros((num_bins, num_iters))

def height_to_color(z, zmin=0, zmax=10):
    z = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    colors = np.zeros((z.shape[0], z.shape[1], 4))
    colors[..., 0] = np.clip(4 * z - 2, 0, 1)
    colors[..., 1] = np.clip(4 * z, 0, 1)
    colors[..., 2] = np.clip(4 * (1 - z), 0, 1)
    colors[..., 3] = 0.8
    return colors

try:
    mesh = gl.GLSurfacePlotItem(x=x, y=y, z=Z, shader='balloon', colors=height_to_color(Z), smooth=True, computeNormals=False, glOptions='translucent')
    mesh.setGLOptions('translucent')
    view.addItem(mesh)
except Exception as e:
    print(f"Warning: Failed to use 'balloon' shader: {e}. Falling back to 'shaded'.")
    mesh = gl.GLSurfacePlotItem(x=x, y=y, z=Z, shader='shaded', colors=height_to_color(Z), smooth=True, computeNormals=False, glOptions='translucent')
    mesh.setGLOptions('translucent')
    view.addItem(mesh)

# Animation loop
iter_idx = [0]
animating = [True]
timer = QtCore.QTimer()
current_ref_idx = [0]

def update():
    if not animating[0]:
        return
    idx = iter_idx[0]
    Z[:, :-1] = Z[:, 1:].copy()
    Z[:, -1] = fft_array_norm[idx].copy()
    mesh.setData(z=Z, colors=height_to_color(Z))
    loss_text.setData(text=f'Total Loss: {total_power_loss_db:.2f} dB')
    gain_text.setData(text=f'Total Gain: {total_gain_db:.2f} dB')
    echo_gain_text.setData(text=f'Echo Gain: {echo_gain_db:.2f} dB')
    efficiency = 100 * np.exp(total_gain_db / 20 - total_power_loss_db / 20)
    efficiency_text.setData(text=f'Efficiency: {efficiency:.1f}%')
    ref_loss = loss_per_reflection[current_ref_idx[0] % num_reflections]
    ref_loss_text.setData(text=f'Ref Loss (R{current_ref_idx[0] % num_reflections + 1}): {ref_loss:.2f} dB')
    constants_text.setData(text=f'Constants: h={h:.2e}, G={G:.2e}')
    current_ref_idx[0] += 1
    iter_idx[0] = (idx + 1) % num_iters

timer.timeout.connect(update)
timer.start(30)

# Keyboard controls
def keyPressEvent(event):
    try:
        if event.key() == QtCore.Qt.Key_Space:
            animating[0] = not animating[0]
            if animating[0]:
                timer.start(30)
                print("Animation resumed.")
            else:
                timer.stop()
                print("Animation paused.")
        elif event.key() == QtCore.Qt.Key_W:
            view.opts['center'] += pg.Vector(0, -10, 0)
            print("Panning up (W)")
            view.update()
        elif event.key() == QtCore.Qt.Key_S:
            view.opts['center'] += pg.Vector(0, 10, 0)
            print("Panning down (S)")
            view.update()
        elif event.key() == QtCore.Qt.Key_A:
            view.opts['center'] += pg.Vector(-10, 0, 0)
            print("Panning left (A)")
            view.update()
        elif event.key() == QtCore.Qt.Key_D:
            view.opts['center'] += pg.Vector(10, 0, 0)
            print("Panning right (D)")
            view.update()
        elif event.key() == QtCore.Qt.Key_R:
            view.opts['center'] = pg.Vector(f_end/2, num_iters/2, 5)
            view.opts['distance'] = 400
            view.opts['elevation'] = 45
            view.opts['azimuth'] = 60
            view.setCameraParams(distance=400, fov=60, elevation=45, azimuth=60)
            print("View reset (R)")
            view.update()
        elif event.key() == QtCore.Qt.Key_P:
            custom_panning[0] = not custom_panning[0]
            if custom_panning[0]:
                view.mousePressEvent = customMousePressEvent
                view.mouseMoveEvent = customMouseMoveEvent
                print("Custom panning enabled (right-drag).")
            else:
                view.mousePressEvent = None
                view.mouseMoveEvent = None
                print("Custom panning disabled, reverting to default controls.")
    except Exception as e:
        print(f"Error in keyPressEvent: {e}")

view.keyPressEvent = keyPressEvent

# Run
if __name__ == '__main__':
    try:
        print("Hardware acceleration enabled via QSurfaceFormat (Compatibility Profile).")
        print("Mouse controls: Left drag = orbit/rotate, Right drag = pan/move, Wheel = zoom.")
        print("Keyboard controls: Spacebar = pause/resume animation, WASD = pan (up/down/left/right), R = reset view, P = toggle custom panning.")
        print(f"Loaded {num_iters} time steps, {num_bins} frequency bins.")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)