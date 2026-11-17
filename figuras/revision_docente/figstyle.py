# Shared style of every figure of the article and of the supplement.
# One typeface, one set of sizes, one palette with fixed meanings, and one page width: every figure is drawn 16 cm wide
# and included at \textwidth, so the lettering has the same size in all of them.
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W = 6.30                       # inches = 16 cm, the text width of the article
LANG = os.environ.get('FIGLANG', 'es')            # language of the labels: es (Spanish manuscript) or en (submission version)
def tr(en, es): return es if LANG == 'es' else en
OUT = os.environ.get('FIGOUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '')) + LANG + '/'
os.makedirs(OUT, exist_ok=True)

# neutrals
INK, GREY, MID, LIGHT, PALE, RULE, CARD = '#1f1f1d', '#52514e', '#8b8a85', '#c9c8c2', '#ecebe6', '#dddcd6', '#f6f5f1'
# hues, each with one meaning throughout the article
ORANGE = '#eb6834'             # duplicates, leakage, naive protocol
CHAR = '#3a3936'               # grouped protocol (same rows, groups kept together)
BLUE = '#2a78d6'               # distinct images, deduplicated protocol
GREEN = '#12925f'              # colour-feature models
PURPLE = '#6250d6'             # convolutional networks
RED = '#e34948'                # anaemia (clinical class)
TINT = {ORANGE: '#fbe3d8', BLUE: '#dbe8f8', GREEN: '#d9efe6', PURPLE: '#e5e1f8', CHAR: '#e4e3df', RED: '#fadcdc'}

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Liberation Sans', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MID, 'axes.linewidth': 0.6,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': GREY, 'ytick.color': GREY,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'grid.color': PALE, 'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.dpi': 600,
    'lines.solid_capstyle': 'round', 'hatch.linewidth': 0.5, 'axes.axisbelow': True})

FS, FS_S, FS_XS, FS_L = 8, 7.5, 7, 10        # body, small, smallest, panel letter


def figure(h):
    """Figure 16 cm wide and h inches tall. Positions are then given in inches from the top-left corner."""
    fig = plt.figure(figsize=(W, h))
    fig._H = h
    return fig


def axes(fig, x, y, w, h, **kw):
    """Axes whose top-left corner is x inches from the left and y inches from the top."""
    H = fig._H
    return fig.add_axes([x / W, (H - y - h) / H, w / W, h / H], **kw)


def canvas(fig):
    """Invisible axes covering the whole figure, in inches, y pointing down: for drawings and free text."""
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_xlim(0, W); ax.set_ylim(fig._H, 0); ax.axis('off')
    return ax


def text(fig, x, y, s, **kw):
    kw.setdefault('va', 'top'); kw.setdefault('ha', 'left'); kw.setdefault('fontsize', FS)
    return fig.text(x / W, 1 - y / fig._H, s, **kw)


def panel(fig, x, y, letter, title=None, **kw):
    """Bold panel letter with an optional title on the same line."""
    text(fig, x, y, letter, fontsize=FS_L, fontweight='bold')
    if title:
        text(fig, x + 0.19, y + 0.022, title, fontsize=FS, **kw)


def card(c, x, y, w, h, fc=CARD, ec='none', lw=0.6, r=0.06, z=0):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, fc=fc, ec=ec, lw=lw, zorder=z)
    c.add_patch(p); return p


def arrow(c, x0, y0, x1, y1, color=MID, lw=0.9, ms=7, z=2, style='-|>', **kw):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z, **kw)
    c.add_patch(a); return a


def clean(ax, keep=('left', 'bottom')):
    for s in ax.spines:
        ax.spines[s].set_visible(s in keep)


def save(fig, name, check=True):
    """Save at the exact size (no tight box) after checking that nothing is drawn outside the page."""
    fig.canvas.draw()
    if check:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())          # inches
        tol = 0.012
        assert bb.x0 >= -tol and bb.y0 >= -tol and bb.x1 <= W + tol and bb.y1 <= fig._H + tol, \
            '%s: content outside the page: %s (page %.2f x %.2f)' % (name, np.round([bb.x0, bb.y0, bb.x1, bb.y1], 3), W, fig._H)
    fig.savefig(OUT + name + '.pdf'); fig.savefig(OUT + name + '.png', dpi=200); plt.close(fig)
    print('saved', name, 'height %.2f in' % fig._H)
