"""Visualization functions for UTD field results"""

import numpy as np

from ..utils import edge_xy, scalar


# -----------------------------------------------------------------------------
# Shared plotting functions for gradient demos
# -----------------------------------------------------------------------------

def draw_edges(ax, edges, color='k', lw=2, alpha=0.7):
    """Draw edges on axes.

    Args:
        ax: matplotlib axes
        edges: list of Edge2D objects
        color: line color
        lw: line width
        alpha: transparency
    """
    for edge in edges:
        p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
        ax.plot([p0_x, p1_x], [p0_y, p1_y], color=color, lw=lw, alpha=alpha)


def draw_edges_with_normals(ax, edges, color='k', lw=2, alpha=0.7,
                             normal_scale=0.3, normal_color='lime'):
    """Draw edges with normal arrows.

    Args:
        ax: matplotlib axes
        edges: list of Edge2D objects
        color: line color
        lw: line width
        alpha: transparency
        normal_scale: scale factor for normal arrows
        normal_color: color for normal arrows
    """
    for edge in edges:
        p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
        ax.plot([p0_x, p1_x], [p0_y, p1_y], color=color, lw=lw, alpha=alpha)

        # Draw normal arrow at edge midpoint
        mid_x, mid_y = (p0_x + p1_x) / 2, (p0_y + p1_y) / 2
        n_x, n_y = scalar(edge.normal.x), scalar(edge.normal.y)
        ax.arrow(mid_x, mid_y, n_x * normal_scale, n_y * normal_scale,
                 head_width=0.15, head_length=0.1, fc=normal_color, ec='black',
                 zorder=6, alpha=alpha)


def draw_corners(ax, corners, marker_color='r', marker_size=8,
                 face0_color='cyan', facen_color='magenta', alpha=0.8):
    """Draw corners with face direction arrows.

    Args:
        ax: matplotlib axes
        corners: list of Corner2D objects
        marker_color: corner marker color
        marker_size: corner marker size
        face0_color: color for face0 direction arrow
        facen_color: color for face_n direction arrow
        alpha: transparency
    """
    for corner in corners:
        c_x = scalar(corner.position.x)
        c_y = scalar(corner.position.y)
        f0_x = scalar(corner.face0_point.x)
        f0_y = scalar(corner.face0_point.y)
        fn_x = scalar(corner.face_n_point.x)
        fn_y = scalar(corner.face_n_point.y)

        # Draw corner point
        ax.plot(c_x, c_y, 'o', color=marker_color, markersize=marker_size,
                alpha=0.7, zorder=7)

        # Face0 direction arrow
        dir0_x, dir0_y = f0_x - c_x, f0_y - c_y
        ax.arrow(c_x, c_y, dir0_x, dir0_y,
                 head_width=0.1, head_length=0.08, fc=face0_color, ec='blue',
                 zorder=8, alpha=alpha, linewidth=1.5)

        # Face_n direction arrow
        dirn_x, dirn_y = fn_x - c_x, fn_y - c_y
        ax.arrow(c_x, c_y, dirn_x, dirn_y,
                 head_width=0.1, head_length=0.08, fc=facen_color, ec='purple',
                 zorder=8, alpha=alpha, linewidth=1.5)


def draw_tx(ax, tx_pos, color='white', marker='*', size=100, edgecolor='black'):
    """Draw transmitter marker on axes.

    Args:
        ax: matplotlib axes
        tx_pos: tuple (x, y) or (x, y, z)
        color: marker fill color
        marker: marker style
        size: marker size
        edgecolor: marker edge color
    """
    tx_x = tx_pos[0]
    tx_y = tx_pos[1]
    ax.scatter([tx_x], [tx_y], c=color, s=size, marker=marker,
               edgecolors=edgecolor, zorder=5)


def draw_scene(ax, edges, tx_pos, range_x, range_y):
    """Draw scene with edges and transmitter.

    Args:
        ax: matplotlib axes
        edges: list of Edge2D objects
        tx_pos: tuple (x, y) or (x, y, z)
        range_x: tuple (x_min, x_max)
        range_y: tuple (y_min, y_max)
    """
    draw_tx(ax, tx_pos)
    draw_edges(ax, edges)
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_aspect('equal')


def plot_field_with_edges(ax, data, title, edges, tx_pos, range_x, range_y,
                          vmin, vmax, cmap='jet'):
    """Plot field heatmap with edges and transmitter.

    Args:
        ax: matplotlib axes
        data: 2D numpy array
        title: plot title
        edges: list of Edge2D objects
        tx_pos: tuple (x, y) or (x, y, z)
        range_x, range_y: plot bounds
        vmin, vmax: colorbar limits
        cmap: colormap

    Returns:
        im: matplotlib image object
    """
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax)
    draw_scene(ax, edges, tx_pos, range_x, range_y)
    ax.set_title(title, fontsize=9)
    return im


def plot_gradient_with_edges(ax, data, title, edges, tx_pos, range_x, range_y,
                             vmin=None, vmax=None):
    """Plot gradient field with symmetric colorbar.

    Args:
        ax: matplotlib axes
        data: 2D numpy array
        title: plot title
        edges: list of Edge2D objects
        tx_pos: tuple (x, y) or (x, y, z)
        range_x, range_y: plot bounds
        vmin, vmax: colorbar limits (if None, uses symmetric auto-scaling)

    Returns:
        im: matplotlib image object
    """
    if vmin is None or vmax is None:
        abs_max = np.abs(data).max()
        if abs_max < 1e-10:
            abs_max = 1
        vmin, vmax = -abs_max, abs_max

    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    im = ax.imshow(data, extent=extent, origin='lower', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax)
    draw_tx(ax, tx_pos, color='black')
    draw_edges(ax, edges)
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    return im


__all__ = [
    "draw_edges",
    "draw_edges_with_normals",
    "draw_corners",
    "draw_tx",
    "draw_scene",
    "plot_field_with_edges",
    "plot_gradient_with_edges",
]
