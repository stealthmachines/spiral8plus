#!/usr/bin/env python3
"""
φ-Spiral Visualization Helper
Visualize output from FASTA DNA Unified Framework V1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_spiral_data(filename='spiral_output.csv'):
    """Load spiral data from CSV"""
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} points from {filename}")

    # Split strands
    strand1 = df[df['strand'] == 1]
    strand2 = df[df['strand'] == 2]

    print(f"  Strand 1: {len(strand1)} points")
    print(f"  Strand 2: {len(strand2)} points")

    return df, strand1, strand2

def plot_3d_spiral(strand1, strand2, title='DNA φ-Spiral (Unified Framework)'):
    """Create 3D plot of double helix"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot strands
    ax.plot(strand1['x'], strand1['y'], strand1['z'],
            color='cyan', alpha=0.7, linewidth=1.0, label='Strand 1')
    ax.plot(strand2['x'], strand2['y'], strand2['z'],
            color='magenta', alpha=0.7, linewidth=1.0, label='Strand 2')

    # Labels
    ax.set_xlabel('X (φ-scaled)', fontsize=12)
    ax.set_ylabel('Y (φ-scaled)', fontsize=12)
    ax.set_zlabel('Z (helical)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        strand1['x'].max()-strand1['x'].min(),
        strand1['y'].max()-strand1['y'].min(),
        strand1['z'].max()-strand1['z'].min()
    ]).max() / 2.0

    mid_x = (strand1['x'].max()+strand1['x'].min()) * 0.5
    mid_y = (strand1['y'].max()+strand1['y'].min()) * 0.5
    mid_z = (strand1['z'].max()+strand1['z'].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig, ax

def plot_geometry_distribution(df):
    """Plot geometry distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Strand 1 only (same for strand 2)
    strand1 = df[df['strand'] == 1]

    # Geometry counts
    geom_counts = strand1['geometry'].value_counts()
    colors_map = {
        'Point': 'red',
        'Line': 'green',
        'Triangle': 'violet',
        'Tetrahedron': 'mediumpurple',
        'Pentachoron': 'blue',
        'Hexacross': 'indigo',
        'Heptacube': 'purple',
        'Octacube': 'white'
    }
    colors = [colors_map.get(g, 'gray') for g in geom_counts.index]

    geom_counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
    ax1.set_title('Geometry Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Geometry Type')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)

    # Base counts
    base_counts = strand1['base'].value_counts()
    base_counts.plot(kind='bar', ax=ax2, color=['red', 'green', 'blue', 'orange'],
                     edgecolor='black')
    ax2.set_title('DNA Base Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Base')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    return fig

def plot_amplitude_evolution(strand1):
    """Plot amplitude evolution along genome"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Amplitude vs index
    ax1.plot(strand1['index'], strand1['amplitude'],
             color='purple', alpha=0.6, linewidth=0.5)
    ax1.set_title('D_n Operator Amplitude Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Genome Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # Phase vs index
    ax2.scatter(strand1['index'], strand1['phase'],
                c=strand1['dimension'], cmap='rainbow',
                s=1, alpha=0.5)
    ax2.set_title('Phase Evolution (colored by dimension)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Genome Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Dimension')

    plt.tight_layout()
    return fig

def export_frames_for_animation(strand1, strand2, output_dir='frames', num_frames=360):
    """Export rotation frames for video"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting {num_frames} frames to {output_dir}/...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_frames):
        ax.clear()

        # Plot strands
        ax.plot(strand1['x'], strand1['y'], strand1['z'],
                color='cyan', alpha=0.7, linewidth=1.0)
        ax.plot(strand2['x'], strand2['y'], strand2['z'],
                color='magenta', alpha=0.7, linewidth=1.0)

        # Rotate view
        angle = i * 360.0 / num_frames
        ax.view_init(elev=15, azim=angle)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'DNA φ-Spiral (Frame {i+1}/{num_frames})')

        # Save frame
        filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')

        if (i + 1) % 30 == 0:
            print(f"  Exported {i+1}/{num_frames} frames")

    plt.close(fig)
    print(f"\nFrames saved to {output_dir}/")
    print(f"Create video with: ffmpeg -r 30 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p spiral_animation.mp4")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_spiral.py <spiral_output.csv> [--animate]")
        print("\nOptions:")
        print("  --animate    Export 360 frames for video creation")
        return

    filename = sys.argv[1]
    animate = '--animate' in sys.argv

    # Load data
    df, strand1, strand2 = load_spiral_data(filename)

    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total points: {len(strand1)}")
    print(f"Amplitude range: {strand1['amplitude'].min():.6e} to {strand1['amplitude'].max():.6e}")
    print(f"Mean amplitude: {strand1['amplitude'].mean():.6e}")
    print(f"\nGeometry distribution:")
    print(strand1['geometry'].value_counts().to_string())

    # Create visualizations
    print("\nGenerating plots...")

    # 3D spiral
    fig1, ax1 = plot_3d_spiral(strand1, strand2)

    # Geometry distribution
    fig2 = plot_geometry_distribution(df)

    # Amplitude evolution
    fig3 = plot_amplitude_evolution(strand1)

    # Animation frames
    if animate:
        export_frames_for_animation(strand1, strand2)

    print("\nDisplaying plots...")
    plt.show()

if __name__ == '__main__':
    main()
