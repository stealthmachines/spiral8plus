import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

phi = (1 + np.sqrt(5)) / 2
golden_angle = 360 / phi**2  # 137.507°
colors = ['red', 'green', 'violet', 'ultraviolet', 'ultraviolet', 'ultraviolet', 'ultraviolet', 'vacuum ultraviolet']  # Adjusted to framework
alpha_vals = [0.015269, 0.008262, 0.110649, -0.083485, 0.025847, -0.045123, 0.067891, 0.012345]
growth_rates = [np.exp(abs(a)) for a in alpha_vals]  # From framework spirals
periods = [13.057] * 8  # Consistent period
vertex_counts = [1, 2, 3, 4, 5, 12, 128, 256]  # From polytopes

def generate_spiral_points(dim, num_points, alpha, growth, period):
    theta = np.linspace(0, period * 2 * np.pi, num_points)
    r = growth ** (theta / (2 * np.pi))
    # Multi-dim coords (project higher to 3D)
    coords = np.zeros((num_points, max(dim, 3)))
    for i in range(num_points):
        angle = theta[i] * golden_angle * np.pi / 180
        coords[i, 0] = r[i] * np.cos(angle)
        coords[i, 1] = r[i] * np.sin(angle)
        if dim >= 3:
            coords[i, 2] = theta[i] / (2 * np.pi) * alpha  # Helical rise
        if dim > 3:
            # Simulate higher dims with orthogonal projection (simple sin/cos modulation)
            for d in range(3, dim):
                coords[i, 0] += np.sin(angle * d) * r[i] / phi  # Fold into x
                coords[i, 1] += np.cos(angle * d) * r[i] / phi  # Fold into y
                coords[i, 2] += np.tanh(alpha * d) * r[i] / phi  # Fold into z, using hyperbolic
    return coords[:, :3]  # Project to 3D for plotting

def update_frame(frame, ax, points_list):
    ax.clear()
    for d in range(1, 5):  # Plot up to 4D for visibility; extend as needed
        points = points_list[d-1]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors[d-1], s=50, alpha=0.8)
        if d > 1:  # Connect to form geometry edges (simple convex hull approx)
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color=colors[d-1], lw=1)
            except:
                pass  # Fallback if hull fails
    ax.view_init(elev=20, azim=frame)
    ax.set_axis_off()
    ax.set_box_aspect([1,1,1])

# Generate points for each dim
points_list = []
for d in range(1, 9):
    points = generate_spiral_points(d, vertex_counts[d-1], alpha_vals[d-1], growth_rates[d-1], periods[d-1])
    points_list.append(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, lambda f: update_frame(f, ax, points_list), frames=360, interval=50)
# ani.save('emerging_geometries_spiral.gif', writer='pillow')
plt.show()