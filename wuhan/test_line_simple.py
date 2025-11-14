"""Ultra-simple VisPy test - just one line"""
from vispy import app, scene
import numpy as np

# Create canvas
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), bgcolor='black', title="Simple Line Test")
view = canvas.central_widget.add_view()

# Create a simple line - just 3 points in a straight line
positions = np.array([
    [0, 0, 0],
    [10, 10, 10],
    [20, 20, 20]
], dtype=np.float32)

# Create Line visual
from vispy.scene.visuals import Line
line = Line(pos=positions, color='cyan', width=5, parent=view.scene)

print("Created line with positions:")
print(positions)
print("Line color: cyan, width: 5")

# Show canvas
canvas.show()
print("Canvas shown - you should see a cyan line from (0,0,0) to (20,20,20)")

app.run()