import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

phi = (1 + np.sqrt(5)) / 2
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white']
alpha = 0.8  # For glow/translucency

def update_frame(frame):
    theta = np.linspace(0, 10 * np.pi, 2000)  # Extended for depth
    r = np.power(phi, theta / (2 * np.pi))  # φ-scaled radius
    z = theta / (2 * np.pi)  # Helical rise, tying to 8D period ~13
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ax.clear()
    segment_len = len(theta) // len(colors)
    for i, color in enumerate(colors):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(colors) - 1 else len(theta)
        ax.plot(x[start:end], y[start:end], z[start:end], color=color, linewidth=3, alpha=alpha)

    # Rotate view for dynamic effect
    ax.view_init(elev=20, azim=frame)
    ax.set_axis_off()
    ax.set_box_aspect([1,1,1])  # Equal scaling

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update_frame, frames=360, interval=50)
# ani.save('phi_3d_spiral.gif', writer='pillow')  # Uncomment for GIF
plt.show()