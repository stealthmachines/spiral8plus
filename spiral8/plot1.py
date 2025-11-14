import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import json

# Fix font issues by setting a fallback font and suppressing Unicode warnings
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# === DATA EXTRACTION ===
phi = (1 + np.sqrt(5)) / 2  # ≈1.618

notes = ['C (n)', 'D (β)', 'E (Ω)', 'F (k)', 'G (Ψ)', 'A (Χ)', 'B (Φ)', 'C (Θ)']
params = ['n', 'β', 'Ω', 'k', 'Ψ', 'Χ', 'Φ', 'Θ']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white']
geometries = ['Point', 'Line', 'Triangle', 'Tetrahedron', 'Pentachoron', 'Hexacross', 'Heptacube', 'Octahedron']
dims = np.arange(1, 9)

# Table 1
alpha_values = [0.015269, 0.008262, 0.110649, -0.083485, 0.025847, -0.045123, 0.067891, 0.012345]
phi_factors = [0.064681, 0.021630, 0.179034, -0.083485, 0.015974, -0.017235, 0.016027, 0.001801]

# Table 2A
freq_ratios = [1.000000, 1.090508, 1.189207, 1.296840, 1.414214, 1.542211, 1.681793, 1.834008]

# Table 2B
harmonics = [1.000, 0.541, 7.247, 5.468, 1.693, 2.955, 4.446, 0.809]

# Table 3A
wavelengths = [700.0, 550.3, 432.6, 340.1, 267.4, 210.2, 165.2, 129.9]
energies = [1.771, 2.253, 2.866, 3.646, 4.638, 5.899, 7.504, 9.545]

# Table 3B
hue_angles = [0.0, 137.5, 275.0, 52.5, 190.0, 327.5, 105.0, 242.6]

# Table 4A
rot_angles = [0.0, 137.5, 275.0, 52.5, 190.0, 327.5, 105.0, 242.6]

# Table 4B
scales = [1.031614, 1.010463, 1.089973, 0.960622, 1.007717, 0.991740, 1.007742, 1.000867]

# Table 5A
spiral_growth = [1.015386, 1.008296, 1.117003, 1.087069, 1.026184, 1.046157, 1.070249, 1.012422]

# Table 6A
vertices = [1, 2, 3, 4, 5, 12, 128, 256]
phi_sym = [1.000000, 1.236068, 1.145898, 0.944272, 0.729490, 1.082039, 7.133196, 8.817115]

# Table 7A
curvature_k = [0.007347, 0.003976, 0.053195, -0.040152, 0.012437, -0.021710, 0.032658, 0.005940]

# Coupling Matrix (8x8)
Gamma = np.array([
    [1.0000, 0.0001, 0.0006, -0.0003, 0.0001, -0.0001, 0.0001, 0.0000],
    [0.0001, 1.0000, 0.0006, -0.0003, 0.0001, -0.0001, 0.0001, 0.0000],
    [0.0006, 0.0006, 1.0000, -0.0057, 0.0011, -0.0012, 0.0011, 0.0001],
    [-0.0003, -0.0003, -0.0057, 1.0000, -0.0013, 0.0014, -0.0013, -0.0002],
    [0.0001, 0.0001, 0.0011, -0.0013, 1.0000, -0.0007, 0.0007, 0.0001],
    [-0.0001, -0.0001, -0.0012, 0.0014, -0.0007, 1.0000, -0.0019, -0.0002],
    [0.0001, 0.0001, 0.0011, -0.0013, 0.0007, -0.0019, 1.0000, 0.0005],
    [0.0000, 0.0000, 0.0001, -0.0002, 0.0001, -0.0002, 0.0005, 1.0000]
])

# === COMPREHENSIVE VISUALIZATION ===
fig = plt.figure(figsize=(24, 30))
fig.suptitle('MUSICAL SCORE: 8 GEOMETRIES + φ SCALING: COMPLETE OCTAVE ANALYSIS', fontsize=18, fontweight='bold', y=0.98)

# 1. Color Spectrum + Wavelength + Energy
ax1 = fig.add_subplot(4, 3, 1)
wavelengths_np = np.array(wavelengths)
norm_wl = (wavelengths_np - min(wavelengths_np)) / (max(wavelengths_np) - min(wavelengths_np))
cmap = LinearSegmentedColormap.from_list("phi_spectrum", colors, N=256)
for i in range(8):
    ax1.barh(i, 1, color=cmap(norm_wl[i]), edgecolor='black', height=0.8)
    ax1.text(1.02, i, f'{wavelengths[i]:.1f} nm', va='center', fontsize=9)
    ax1.text(1.35, i, f'{energies[i]:.3f} eV', va='center', fontsize=9)
ax1.set_yticks(range(8))
ax1.set_yticklabels([f'{p} {n}' for p,n in zip(params, notes)])
ax1.set_xlim(0, 1.6)
ax1.set_title('RAINBOW: COLOR SPECTRUM (λ & Energy)')
ax1.invert_yaxis()
ax1.axis('off')

# 2. φ-Frequency Ratios (Log Scale)
ax2 = fig.add_subplot(4, 3, 2)
ax2.semilogy(dims, freq_ratios, 'o-', color='purple', lw=2, markersize=8)
for i, (r, n) in enumerate(zip(freq_ratios, notes)):
    ax2.text(dims[i], r*1.05, n.split()[0], ha='center', fontsize=9)
ax2.set_xticks(dims)
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Frequency Ratio (×f₀)')
ax2.set_title('MUSIC: φ-HARMONIC FREQUENCY RATIOS')
ax2.grid(True, alpha=0.3)

# 3. Geometric Harmonics
ax3 = fig.add_subplot(4, 3, 3)
bars = ax3.bar(dims, harmonics, color=colors, edgecolor='black')
for bar, h, geo in zip(bars, harmonics, geometries):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{h:.2f}', ha='center', fontsize=8)
    ax3.text(bar.get_x() + bar.get_width()/2, -0.5, geo, ha='center', rotation=45, fontsize=7)
ax3.set_xticks(dims)
ax3.set_ylabel('Harmonic Amplitude')
ax3.set_title('RULER: GEOMETRIC HARMONIC SERIES')
ax3.set_ylim(0, max(harmonics)*1.2)

# 4. φ-Spiral Growth & Rotation
ax4 = fig.add_subplot(4, 3, 4, polar=True)
theta_rad = np.deg2rad(rot_angles)
r = spiral_growth
ax4.scatter(theta_rad, r, c=colors, s=100, edgecolors='black')
for i, (th, rr, p) in enumerate(zip(theta_rad, r, params)):
    ax4.annotate(p, (th, rr), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
ax4.set_title('SPIRAL: φ-SPIRAL GROWTH & ROTATION', pad=20)

# 5. Polytope Vertices & φ-Symmetry
ax5 = fig.add_subplot(4, 3, 5)
ax5t = ax5.twinx()
ax5.bar(dims - 0.2, vertices, width=0.4, label='Vertices', color='lightblue', alpha=0.7)
ax5t.plot(dims + 0.2, phi_sym, 'D-', color='gold', label='φ-Symmetry', markersize=8)
ax5.set_xticks(dims)
ax5.set_ylabel('Vertices', color='blue')
ax5t.set_ylabel('φ-Symmetry Factor', color='gold')
ax5.set_yscale('log')
ax5.set_title('TRIANGLE: POLYTOPE COMPLEXITY')
ax5.set_xticklabels(geometries, rotation=45, ha='right')

# 6. Coupling Matrix Heatmap
ax6 = fig.add_subplot(4, 3, 6)
im = ax6.imshow(Gamma, cmap='coolwarm', vmin=-0.01, vmax=0.01, aspect='equal')
for i in range(8):
    for j in range(8):
        text = ax6.text(j, i, f'{Gamma[i, j]:.4f}', ha="center", va="center", color="black" if abs(Gamma[i, j]) < 0.005 else "white", fontsize=8)
ax6.set_xticks(range(8))
ax6.set_yticks(range(8))
ax6.set_xticklabels(params)
ax6.set_yticklabels(params)
plt.colorbar(im, ax=ax6, label='Γ_ij')
ax6.set_title('GLOBE: 8×8 φ-COUPLING MATRIX')

# 7. α vs φ-Factor Scatter
ax7 = fig.add_subplot(4, 3, 7)
sc = ax7.scatter(alpha_values, phi_factors, c=hue_angles, cmap='hsv', s=120, edgecolors='black')
for i, (a, p, n) in enumerate(zip(alpha_values, phi_factors, params)):
    ax7.annotate(n, (a, p), xytext=(5,5), textcoords='offset points', fontsize=9)
ax7.set_xlabel('α Value')
ax7.set_ylabel('φ-Factor')
ax7.set_title('TARGET: α vs φ-FACTOR (Hue = Rotation)')
plt.colorbar(sc, ax=ax7, label='Rotation Angle °')

# 8. Hyperbolic Curvature
ax8 = fig.add_subplot(4, 3, 8)
ax8.plot(dims, curvature_k, 's-', color='darkred', lw=2, markersize=8)
for i, (k, p) in enumerate(zip(curvature_k, params)):
    ax8.annotate(p, (dims[i], k), xytext=(0, 8 if k > 0 else -12),
                 textcoords='offset points', ha='center', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
ax8.axhline(0, color='black', lw=1, alpha=0.5)
ax8.set_xticks(dims)
ax8.set_ylabel('Curvature κ')
ax8.set_title('CYCLONE: HYPERBOLIC φ-SPACES')
ax8.grid(True, alpha=0.3)

# 9. φ-Scaling Transformations
ax9 = fig.add_subplot(4, 3, 9)
ax9.stem(dims, scales, basefmt=" ")
for i, (s, n) in enumerate(zip(scales, notes)):
    ax9.text(dims[i], s + 0.005 if s > 1 else s - 0.01, f'{n.split()[0]}\n{s:.3f}',
             ha='center', va='bottom' if s > 1 else 'top', fontsize=8)
ax9.axhline(1, color='black', linestyle='--', alpha=0.5)
ax9.set_xticks(dims)
ax9.set_ylabel('Scaling S = φ^exp')
ax9.set_title('ARROWS: φ-SCALING TRANSFORMATIONS')
ax9.set_ylim(min(scales)*0.98, max(scales)*1.02)

# 10. Unified Field Radar
ax10 = fig.add_subplot(4, 3, 10, polar=True)
angles = np.linspace(0, 2*np.pi, 8, endpoint=False).tolist()
angles += angles[:1]
values = np.abs(Gamma.diagonal()).tolist() + [Gamma.diagonal()[0]]
values = [v if v > 0.01 else 0.01 for v in values]  # log scale safe
ax10.plot(angles, values, 'o-', linewidth=2, color='magenta')
ax10.fill(angles, values, alpha=0.25, color='magenta')
ax10.set_xticks(angles[:-1])
ax10.set_xticklabels(params)
ax10.set_title('GALAXY: UNIFIED FIELD DIAGONAL (Self-Coupling)', pad=20)

# 11. Coupling Matrix Line Plot (Alternative to Dendrogram)
ax11 = fig.add_subplot(4, 3, 11)
# Plot off-diagonal elements as coupling strength
for i in range(8):
    coupling_strength = [abs(Gamma[i, j]) for j in range(8) if i != j]
    ax11.plot(range(len(coupling_strength)), coupling_strength, 'o-', label=params[i], alpha=0.7)
ax11.set_xlabel('Coupled Parameter Index')
ax11.set_ylabel('|Coupling Strength|')
ax11.set_title('LINK: COUPLING STRENGTH PATTERNS')
ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 12. 3D φ-Space Projection (Freq, Energy, Scale)
ax12 = fig.add_subplot(4, 3, 12, projection='3d')
xs = np.array(freq_ratios)
ys = np.array(energies)
zs = np.array(scales)
# Use simpler coloring approach
color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black']
for i, (x, y, z, p, c) in enumerate(zip(xs, ys, zs, params, color_list)):
    ax12.scatter(x, y, z, c=c, s=100, edgecolors='black', alpha=0.7)
    ax12.text(x, y, z, p, fontsize=9)
ax12.set_xlabel('Freq Ratio')
ax12.set_ylabel('Energy (eV)')
ax12.set_zlabel('φ-Scale')
ax12.set_title('CUBE: 3D φ-SPACE PROJECTION')

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.4)

# Save figure
plt.savefig('8_geometries_phi_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save data as JSON (convert numpy types to regular Python types)
data = {
    "dimensions": [int(d) for d in dims],
    "parameters": params,
    "notes": notes,
    "colors": colors,
    "geometries": geometries,
    "alpha": [float(a) for a in alpha_values],
    "phi_factor": [float(p) for p in phi_factors],
    "frequency_ratios": [float(f) for f in freq_ratios],
    "harmonics": [float(h) for h in harmonics],
    "wavelength_nm": [float(w) for w in wavelengths],
    "energy_eV": [float(e) for e in energies],
    "hue_angles": [float(h) for h in hue_angles],
    "rotation_angles": [float(r) for r in rot_angles],
    "scaling": [float(s) for s in scales],
    "spiral_growth": [float(g) for g in spiral_growth],
    "polytope_vertices": [int(v) for v in vertices],
    "phi_symmetry": [float(p) for p in phi_sym],
    "hyperbolic_curvature": [float(k) for k in curvature_k],
    "coupling_matrix": [[float(cell) for cell in row] for row in Gamma.tolist()]
}

with open('eight_geometries_phi_framework.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Visualization saved: 8_geometries_phi_complete_analysis.png")
print("✅ Data saved: eight_geometries_phi_framework.json")
print("🏆 TROPHY: 8 GEOMETRIES + φ = COMPLETE UNIVERSAL HARMONY! 🏆")