# %%
import datetime
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# %%
"""
Developed by H. Miyawaki @ OMU, 2022-2024
"""
# %% set defaults
# sns.set_context('paper')  # fontsize: paper < notebook < talk < poster
# sns.set_style('ticks')
plt.rcParams["font.family"] = "Helvetica"   # default font
plt.rcParams["xtick.direction"] = "out"  # set x-ticks inward/outward
plt.rcParams["ytick.direction"] = "out"  # set y-ticks inward/outward

# The following lines prevent making outline in pdf
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = True    # somehow it crush PdfPages so commented out
# prevent using unicode for "-" on axes
plt.rcParams['axes.unicode_minus'] = False


# %% sample codes
def main():
    import numpy as np

    def panel_a(x, y):
        w = 20
        h = 30
        margin_x = 10
        margin_y = 10

        right = -np.inf
        bottom = -np.inf

        for n in range(9):
            xx = x + (w + margin_x) * int(n % 3)
            yy = y + (h + margin_y) * int(n / 3)
            ax = subplot_mm([xx, yy, w, h])
            right = np.max((right, xx + w))
            bottom = np.max((bottom, yy + h))

            ax.set_title(f"Title {n}")
            ax.plot(np.random.random(100), np.random.random(100), '.')
            box_off()

            if n > 5:
                ax.set_xlabel("X label")
            if n % 3 == 0:
                ax.set_ylabel("Y label")

        return right, bottom

    def panel_b(x, y):
        w = 40
        h = 45
        margin_y = 20
        right = -np.inf
        bottom = -np.inf

        for n in range(2):
            yy = y + (h + margin_y) * n
            ax = subplot_mm([x, yy, w, h])
            right = np.max((right, x + w))
            bottom = np.max((bottom, yy + h))

            im = ax.pcolorfast(np.random.random((100, 100)))
            ax.set_title(f"Title {n}")
            ax.set_xlabel('X label')
            ax.set_ylabel("Y label")
            box_off()

            cax = subplot_mm([x+w+3, yy, 3, h])
            clb = plt.colorbar(im, ax=ax, cax=cax)
            right = np.max((right, x + w + 3))
            bottom = np.max((bottom, yy + h))
            clb.set_label('Scale')

        return right, bottom

    figsize = [180, 135]  # in mm
    upper = True
    fontsize = 7
    label_size = 12
    panel_n = 0

    fig = fig_mm(figsize)
    draw_fig_border()  # remove for final output
    plt.rcParams["font.size"] = fontsize

    x = 15
    y = 10
    right, bottom = panel_a(x, y)
    text_mm([x-10, y-3], alphabet(panel_n, upper=upper), fontsize=label_size)
    panel_n = panel_n+1

    x = right + 20
    y = 10
    panel_b(x, y)
    text_mm([x-10, y-3], alphabet(panel_n, upper=upper), fontsize=label_size)

    add_generator('pyplot_helper.py')  # remove for final output


# %%
def mm_to_inch(mm):
    """
    convert from mm to inch

    Parameters
    ----------
    mm : list
        values in mm
    Returns
    -------
    list
        values in inch
    """

    return [val / 25.4 for val in mm]


# %%
def mm_to_pos(mm, origin='top', fig=None):
    """
    convert mm to relative position on the figure

    Parameters
    ----------
    mm : list with 2 or 4 elements
        specify x, y, width, and height in mm

    origin: str, 'top' or 'bottom', default 'top'
        'top'/'bottom', measure from top/bottom.

    fig: figure handle, default current figure (plt.gcf)

    Returns
    -------
    list
        relative position
    """

    if fig is None:
        fig = plt.gcf()

    wh = fig.get_size_inches() * 25.4
    if len(mm) == 2:
        if origin.lower() == 'top':
            return mm[0] / wh[0], 1 - mm[1] / wh[1]
        else:
            return mm[0] / wh[0], mm[1] / wh[1]
    elif len(mm) == 4:
        if origin.lower() == 'top':
            return mm[0] / wh[0], 1 - (mm[1] + mm[3]) / wh[1], mm[2] / wh[0], mm[3] / wh[1]
        elif origin.lower() == 'bottom':
            return mm[0] / wh[0], mm[1] / wh[1], mm[2] / wh[0], mm[3] / wh[1]
        else:
            print('origin must be "top" or "bottom"')
    else:
        print('mm must have 2 or 4 elements')


# %%
def fig_mm(figsize=[297, 210], **kwargs):
    """
    make pyplot figure with specified size in mm

    Parameters
    ----------
    figsize : list with 2 elements
        specify width, and height in mm

    kwargs: keywords and values
        additional key words and values that will be passed to plt.figure()

    Returns
    -------
    Figure instance
    """
    figsize = mm_to_inch(figsize)
    fig = plt.figure(figsize=figsize, **kwargs)
    return fig


# %%
def subplot_mm(pos_in_mm, origin='top', fig=None, **kwargs):
    """
    add axes, position is specified in mm

    Parameters
    ----------
    pos_in_mm : list with 4 elements
        specify x, y, width, and height

    origin: str, 'top' or 'bottom', default 'top'
        'top'/'bottom', measure from top/bottom.

    fig: figure handle, default current figure (plt.gcf)

    kwargs: keywords and values
        additional key words and values that will be passed to fig.add_axes()

    Returns
    -------
    Axes instance
    """

    if fig is None:
        fig = plt.gcf()
    return fig.add_axes(mm_to_pos(pos_in_mm, origin=origin, fig=fig), **kwargs)


# %%
def text_mm(pos_in_mm, string, origin='top', fig=None, **kwargs):
    """
    add text, position is specified in mm on figure

    Parameters
    ----------
    pos_in_mm : list with 4 elements
        specify x, y, width, and height

    string : str
        plotted text

    origin: str, 'top' or 'bottom', default 'top'
        'top'/'bottom', measure from top/bottom.

    fig: figure handle, default current figure (plt.gcf)

    kwargs: keywords and values
        additional key words and values that will be passed to plt.text()

    Returns
    -------
    Text instance
    """

    if fig is None:
        fig = plt.gcf()
    return plt.text(*mm_to_pos(pos_in_mm, fig=fig, origin=origin), string, transform=fig.transFigure, **kwargs)


# %%
def box_off(ax=None):
    """
    set top and right spines invisible
    (using seaborn.despine() is recommended.
     when you don't want to use seaborn for some reasons, this might be useful)

    Parameters
    ----------
    ax: axis handle, default current axis (plt.gca)

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# %%
def add_generator(filename=[], x=-5, y=-5, fig=None, s=5):
    """
    set top and right spines invisible

    Parameters
    ----------
    x : float,  default -5
    y : float,  default -5
        position of the label from figure edge in mm
        if negative, distance from right/bottom
    fig: figure handle, default current figure (plt.gcf)
    s : float
        font size of the label in points

    Returns
    -------
    Text instance
    """
    if fig is None:
        fig = plt.gcf()

    if x < 0 or y < 0:
        w, h = fig.get_size_inches()
    else:
        w, h = 0, 0

    if filename:
        fname = filename
    else:
        fname = os.path.basename(caller.filename)

    caller = inspect.stack()[1]
    return text_mm([x + w * 25.4 if x < 0 else x, y + h * 25.4 if y < 0 else y],
                   'generated on {today} by {fname}'.format(fname=fname,
                                                            today=datetime.date.today().strftime('%Y-%m-%d')),
                   fontsize=s, horizontalalignment='right' if x < 0 else 'left')

# %%


def draw_fig_border(fig=None, **kwargs):
    """
    add border of fig, mainly for visual check

    Parameters
    ----------
    fig: figure handle, default current figure (plt.gcf)

    kwargs: keywords and values
        additional key words and values that will be passed to fig.add_axes()

    Returns
    -------
    Axes instance
    """

    if fig is None:
        fig = plt.gcf()
    ax = fig.add_axes([0, 0, 1, 1], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def alphabet(num, upper=False):
    """
    convert integer to alphabet

    Parameters
    ----------
    num: integer, thaw will be converted to alphabet (0 > a, 1 > b, ... 25 > z, 26 > aa, 27 > bb ...)

    upper: False for lower case, True for upper case

    Returns
    -------
    string
    """

    times = int(num/26)+1
    num = num % 26
    if upper:
        return chr(num+ord('A'))*times
    else:
        return chr(num+ord('a'))*times


# %%
def align_significance_bars(sig_bars):
    """
    align significance bars

    Parameters
    ----------
    sig_bars: list of list
        list of sig_bars, each sig_bar is a list of [start, end] or [start, end, id]

    Returns

    x: numpy array
        x position of the bars
    y: numpy array
        y position of the bars
    index: numpy array
        id of the bars
    """
    if len(sig_bars) == 0:
        return np.array([]), np.array([]), None

    if len(sig_bars[0]) == 2:
        with_id = False
    else:
        with_id = True

    sig_bars.sort(key=lambda x: x[0])
    stacks = []

    for sig_bar in sig_bars:
        placed = False
        for stack in stacks:
            # check current stack
            if stack[-1][1] <= sig_bar[0]:
                stack.append(sig_bar)
                placed = True
                break
        if not placed:
            # add new stack
            stacks.append([sig_bar])
    x = np.zeros((len(sig_bars), 2))
    y = np.zeros((len(sig_bars), 2))
    index = np.zeros(len(sig_bars))

    n = 0
    for s_idx, stack in enumerate(stacks):
        for pos in stack:
            x[n, :] = np.array(pos[:2])
            y[n, :] = np.array([s_idx, s_idx])
            if with_id:
                index[n] = pos[2]
            n += 1

    if with_id:
        order = np.argsort(index)
        x = x[order]
        y = y[order]
        index = index[order]

        return x, y, index
    else:
        return x, y


# %%


def add_significance_auto(ax, pairs, x_positions, y_positions, significance_text="*", fontsize=10.5,
                          line_width=0.75, base_x_offset=None, gap=None, y_offset=None,
                          min_vertical_length=None, small_gap=None, text_offset=None):
    """
    Add significance annotations to a plot with automatic height adjustment and collision avoidance for horizontal lines.

    Parameters:
    ax : Axes
        Matplotlib Axes object where annotations will be added.
    pairs : list of tuples
        List of tuples representing pairs of indices in x_positions and y_positions that should be annotated
        with significance (e.g., [(0, 1), (2, 3)]).
    x_positions : list or array
        List of x-coordinates of the bars or points to annotate.
    y_positions : list or array
        List of y-coordinates of the bars or points to annotate.
    significance_text : str or list/tuple of str
        The significance text to display (e.g., "*", "**", "ns"). If a list/tuple is provided,
        each text corresponds to a pair in `pairs`.
    fontsize : float
        Font size for the significance text (default: 10.5).
    line_width : float
        Line width for the annotation lines (default: 1.5).
    base_x_offset : float
        Base horizontal offset for avoiding overlapping vertical lines. Defaults to 1% of x-axis range.
    gap : float
        The gap between the bars/points and the annotation line. Defaults to 3% of y-axis range.
    y_offset : float
        Minimum vertical spacing between horizontal lines. Defaults to 3% of y-axis range.
    min_vertical_length : float
        Minimum length of the vertical lines. Defaults to 6% of y-axis range.
    small_gap : float
        Minimum gap between vertical lines and horizontal annotation lines to avoid overlap. Defaults to 1% of y-axis range.
    text_offset : float
        Vertical offset for positioning the text above the horizontal line. Defaults to `gap`.

    Returns:
    bar_y : list
        List of y-coordinates of the horizontal lines.
    """

    if len(pairs) == 0:
        return

    # Get the x- and y-axis ranges
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Set default values based on x- and y-axis ranges if not provided
    if gap is None:
        gap = 0.03 * y_range
    if y_offset is None:
        y_offset = 0.03 * y_range
    if min_vertical_length is None:
        min_vertical_length = 0.06 * y_range
    if small_gap is None:
        small_gap = 0.01 * y_range
    if text_offset is None:
        text_offset = 0.03 * y_range
    if base_x_offset is None:
        base_x_offset = 0.01 * x_range

    vertical_count = {}
    for i, j in pairs:
        for k in [i, j]:
            if x_positions[k] in vertical_count.keys():
                vertical_count[x_positions[k]] += 1
            else:
                vertical_count[x_positions[k]] = 1
    # Track used horizontal lines to avoid overlap (start_x, end_x, y)
    horizontal_lines = []
    vertical_offsets = {
        x: -(vertical_count[x]-1)/2 for x in vertical_count.keys()}

    # Handle significance_text as a list/tuple or a single string
    if isinstance(significance_text, (list, tuple)):
        if len(significance_text) != len(pairs):
            raise ValueError(
                "The length of significance_text must match the number of pairs.")
        texts = significance_text
    else:
        texts = [significance_text] * len(pairs)

    # Process each pair
    bar_y=[]
    for idx, (i, j) in enumerate(pairs):
        # Get x and y positions for the pair
        x1, x2 = x_positions[i], x_positions[j]
        y1, y2 = y_positions[i], y_positions[j]
        base_height = max(y1, y2) + gap

        # Apply per-x_position vertical offsets for minimum x_offset
        x1_pos = x1 + vertical_offsets[x1] * base_x_offset
        x2_pos = x2 + vertical_offsets[x2] * base_x_offset

        # Update vertical offsets for the next vertical lines
        vertical_offsets[x1] += 1
        vertical_offsets[x2] += 1

        # Ensure horizontal line avoids intermediate bars
        intermediate_bars = [
            y_positions[k] for k in range(len(x_positions)) if x1_pos < x_positions[k] < x2_pos
        ]
        if intermediate_bars:
            max_intermediate_height = max(intermediate_bars)
            if base_height <= max_intermediate_height + gap:
                base_height = max_intermediate_height + gap + min_vertical_length

        # Ensure minimum vertical line length
        if base_height - max(y1, y2) < min_vertical_length:
            base_height = max(y1, y2) + min_vertical_length

        # Adjust base height only if lines truly intersect
        def lines_intersect(start1, end1, start2, end2):
            """Check if two horizontal lines overlap in the x-direction."""
            return not (end1 < start2 or end2 < start1)

        for h_start, h_end, h_y in horizontal_lines:
            if lines_intersect(x1_pos, x2_pos, h_start, h_end) and abs(base_height - h_y) < y_offset:
                base_height = h_y + y_offset

        # Draw vertical lines avoiding collision with horizontal lines
        def draw_vertical_line(x_pos, y_start, y_end):
            segments = [(y_start, y_end)]

            for h_start, h_end, horizontal_y in horizontal_lines:
                if h_start <= x_pos <= h_end:  # x位置が水平線の範囲内にある場合
                    new_segments = []
                    for seg_start, seg_end in segments:
                        if seg_start < horizontal_y - small_gap and seg_end > horizontal_y + small_gap:
                            new_segments.append((seg_start, horizontal_y - small_gap))
                            new_segments.append((horizontal_y + small_gap, seg_end))
                        else:
                            new_segments.append((seg_start, seg_end))
                    segments = new_segments
            for seg_start, seg_end in segments:
                ax.plot([x_pos, x_pos], [seg_start, seg_end],
                        color="black", lw=line_width, zorder=3)

        # Draw vertical lines for both points in the pair
        draw_vertical_line(x1_pos, y1 + gap, base_height)
        draw_vertical_line(x2_pos, y2 + gap, base_height)

        # Draw the horizontal line
        ax.plot([x1_pos, x2_pos], [base_height, base_height],
                color="black", lw=line_width, zorder=2)

        # Record the horizontal line information
        horizontal_lines.append((x1_pos, x2_pos, base_height))

        # Add the significance text above the horizontal line
        ax.text((x1_pos + x2_pos) / 2, base_height + text_offset, texts[idx],
                ha="center", va="bottom", color="black", fontsize=fontsize, zorder=4)
        bar_y.append(base_height)
    
    return bar_y


# %%
if __name__ == '__main__':
    main()
