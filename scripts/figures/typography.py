"""Portable serif typography; opt in to TeX for manuscript-identical fonts."""
import re
import shutil

USE_TEX = False


def set_use_tex(enabled=False):
    global USE_TEX
    USE_TEX = bool(enabled)
    if USE_TEX and not all(shutil.which(name) for name in ('latex', 'dvipng')):
        raise RuntimeError('--tex requires latex and dvipng on PATH, plus type1cm and cm-super.')


def configure_plot_fonts(font_size=8):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.size': font_size,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'] if USE_TEX else ['DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'text.usetex': USE_TEX,
        'text.latex.preamble': r'\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}',
        'pdf.fonttype': 42,
        'svg.fonttype': 'path',
        'svg.hashsalt': 'safeprocess-figures',
    })


def method_label(name):
    return r'\textit{' + name + '}' if USE_TEX else r'$\mathit{' + name.replace(' ', r'\ ') + '}$'


def plain_text(value):
    if USE_TEX:
        return value
    # Keep inline mathematics, while removing LaTeX-only text formatting.
    return (value.replace(r'\%', '%').replace(r'\qquad', '    ').replace(r'\quad', '  ')
            .replace(r'\land', r'\wedge').replace(r'\lor', r'\vee'))


def save_figure(fig, path, *, svg=False, dpi=200):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extensions = ['pdf', 'png'] + (['svg'] if svg else [])
    for extension in extensions:
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {'Date': None} if extension == 'svg' else None
        fig.savefig(path.with_suffix('.' + extension), dpi=dpi,
                    bbox_inches='tight', metadata=metadata)
