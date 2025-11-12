"""Minimal VisPy test to verify OpenGL is working"""
from vispy import app, scene
import numpy as np

# Create canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), bgcolor='black', title="Test")
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.distance = 50
view.camera.elevation = 30
view.camera.azimuth = 45

# Create simple test data - a cube
positions = np.array([
    [10, 10, 10],
    [-10, 10, 10],
    [-10, -10, 10],
    [10, -10, 10],
    [10, 10, -10],
    [-10, 10, -10],
    [-10, -10, -10],
    [10, -10, -10],
], dtype=np.float32)

colors = np.array([
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
    [1, 1, 1],  # White
    [0.5, 0.5, 0.5],  # Gray
], dtype=np.float32)

# Create markers
markers = scene.visuals.Markers()
markers.set_data(positions, face_color=colors, size=20, edge_width=0)
markers.parent = view.scene

print("Showing test visualization...")
print(f"Positions shape: {positions.shape}")
print(f"Colors shape: {colors.shape}")
print(f"Camera distance: {view.camera.distance}")

canvas.show()
app.run()
