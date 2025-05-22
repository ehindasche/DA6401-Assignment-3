from libraries import *
from heatmaps import *

def animate_attention(sample, font_prop=None):
    source = list(sample['source'])
    target = list(sample['prediction'])
    attention = sample['attention']  # Shape: [target_len, source_len]

    fig_width = max(2.5, len(source) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 1.8))
    ax.axis('off')

    ax.set_xlim(-0.5, len(source) - 0.5)
    ax.set_ylim(0, 1)

    # Normalize attention weights for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('YlOrBr')

    # Display source characters with background boxes
    text_objs = []
    for i, char in enumerate(source):
        txt = ax.text(i, 0.5, char, fontsize=14, ha='center', va='center',
                      weight='bold', fontproperties=font_prop,
                      bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.25'))
        text_objs.append(txt)

    # Dynamic title for target char
    title = ax.text(0.5, 0.92, '', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=14, fontproperties=font_prop, weight='bold')

    def update(frame):
        weights = attention[frame]
        tgt_char = target[frame]
        title.set_text(f"Generating: '{tgt_char}'")

        for i, weight in enumerate(weights):
            color = cmap(norm(weight))
            text_objs[i].set_bbox(dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.25'))

        return text_objs + [title]

    ani = FuncAnimation(fig, update, frames=len(target), interval=1000, blit=True, repeat=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
from IPython.display import HTML, display

# Show attention animations for first 4 predictions
for i in range(4):
    print(f"Sample {i + 1}:")
    display(animate_attention(test_metrics['predictions'][i]))