import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


def box(ax, x, y, w, h, text, fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fill=False,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )


def setup(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=15)


# ============================================================
# 1. OVERALL METHODOLOGY PIPELINE
# ============================================================

fig, ax = plt.subplots(figsize=(14, 4.5))
setup(ax, "Composite-Noise QKD Simulation Pipeline")

labels = [
    "Protocol-specific\nstate preparation",
    "Sequential noise cascade\nNdep → Ndeph → NAD",
    "Device imperfections\nDark counts + misalignment",
    "Measurement\nand detection",
    "Basis reconciliation\nand sifting",
    "QBER, key rate R\nand PRI",
    "Statistical analysis\n50 trials + bootstrap CI",
]

xs = [0.2, 1.65, 3.45, 5.25, 6.85, 8.25, 9.55]
widths = [1.25, 1.55, 1.45, 1.25, 1.15, 1.15, 1.25]

# Because the last box would extend beyond the plot,
# use a slightly more compact layout.
xs = [0.15, 1.55, 3.25, 4.95, 6.25, 7.55, 8.75]
widths = [1.15, 1.45, 1.4, 1.05, 1.05, 1.05, 1.1]

for x, w, label in zip(xs, widths, labels):
    box(ax, x, 1.9, w, 1.15, label, fontsize=8)

for i in range(len(xs) - 1):
    arrow(
        ax,
        xs[i] + widths[i],
        2.475,
        xs[i + 1],
        2.475,
    )

plt.tight_layout()
plt.savefig(
    "methodology_pipeline.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 2. BB84 SCHEMATIC
# ============================================================

fig, ax = plt.subplots(figsize=(12, 4))
setup(ax, "BB84 Simulation Architecture")

box(ax, 0.3, 2, 1.5, 1, "|0⟩ / |1⟩\nor\n|+⟩ / |−⟩")
box(ax, 2.2, 2, 1.5, 1, "Alice\nbasis choice")
box(ax, 4.2, 1.8, 2.0, 1.4,
    "Composite noise\nNdep → Ndeph → NAD")
box(ax, 6.7, 2, 1.5, 1, "Bob\nbasis choice")
box(ax, 8.7, 2, 1.2, 1, "Measure")

arrow(ax, 1.8, 2.5, 2.2, 2.5)
arrow(ax, 3.7, 2.5, 4.2, 2.5)
arrow(ax, 6.2, 2.5, 6.7, 2.5)
arrow(ax, 8.2, 2.5, 8.7, 2.5)

ax.text(
    5.2,
    1.25,
    "Matching bases are retained during sifting",
    ha="center",
    fontsize=10,
)

plt.tight_layout()
plt.savefig(
    "bb84_schematic.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 3. B92 SCHEMATIC
# ============================================================

fig, ax = plt.subplots(figsize=(12, 4))
setup(ax, "B92 Simulation Architecture")

box(ax, 0.4, 2, 1.5, 1,
    "|0⟩ or |+⟩")
box(ax, 2.4, 2, 2.0, 1,
    "Composite noise\nNdep → Ndeph → NAD")
box(ax, 5.0, 2, 1.8, 1,
    "Projection\n{|1⟩, |−⟩}")
box(ax, 7.4, 2, 1.5, 1,
    "Conclusive\noutcomes")
box(ax, 9.3, 2, 0.7, 1,
    "Key")

arrow(ax, 1.9, 2.5, 2.4, 2.5)
arrow(ax, 4.4, 2.5, 5.0, 2.5)
arrow(ax, 6.8, 2.5, 7.4, 2.5)
arrow(ax, 8.9, 2.5, 9.3, 2.5)

plt.tight_layout()
plt.savefig(
    "b92_schematic.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 4. E91 SCHEMATIC
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
setup(ax, "E91 Simulation Architecture")

# Bell source
circle = Circle(
    (1.5, 2.5),
    0.55,
    fill=False,
    linewidth=1.5,
)
ax.add_patch(circle)

ax.text(
    1.5,
    2.5,
    "Bell\nsource",
    ha="center",
    va="center",
    fontsize=9,
)

# Alice branch
box(ax, 3.0, 3.25, 1.7, 0.9,
    "Alice\nRy(θA)")
box(ax, 5.4, 3.25, 1.7, 0.9,
    "Alice\nmeasurement")

# Bob branch
box(ax, 3.0, 1.0, 1.7, 0.9,
    "Composite noise\nNdep → Ndeph → NAD")
box(ax, 5.4, 1.0, 1.7, 0.9,
    "Bob\nRy(θB)")
box(ax, 7.7, 1.0, 1.7, 0.9,
    "Bob\nmeasurement")

# CHSH
box(ax, 7.7, 3.25, 1.7, 0.9,
    "CHSH\nparameter S")

# Source to Alice
arrow(ax, 2.05, 2.8, 3.0, 3.7)

# Source to Bob/noise
arrow(ax, 2.05, 2.2, 3.0, 1.45)

# Alice
arrow(ax, 4.7, 3.7, 5.4, 3.7)
arrow(ax, 7.1, 3.7, 7.7, 3.7)

# Bob
arrow(ax, 4.7, 1.45, 5.4, 1.45)
arrow(ax, 7.1, 1.45, 7.7, 1.45)

ax.text(
    5.2,
    4.65,
    "Alice and Bob independently select measurement settings",
    ha="center",
    fontsize=9,
)

plt.tight_layout()
plt.savefig(
    "e91_schematic.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Generated:")
print("  methodology_pipeline.png")
print("  bb84_schematic.png")
print("  b92_schematic.png")
print("  e91_schematic.png")
