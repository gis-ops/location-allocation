from matplotlib import pyplot as plt
from itertools import cycle


def plot_result(points, point_allocations, opt_sites):
    """
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
    """
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c="black", s=4)
    ax = plt.gca()

    plt.scatter(opt_sites[:, 0], opt_sites[:, 1], c="C1", s=200, marker="*")

    cycol = cycle("bgrcmk")

    for k, v in point_allocations.items():
        color = next(cycol)
        for point_idx in v:
            plt.scatter(points[point_idx][0], points[point_idx][1], c=color, marker="+")

    # for site in opt_sites:
    #     circle = plt.Circle(site, radius, color='C1',fill=False,lw=2)
    #     ax.add_artist(circle)
    ax.axis("equal")
    ax.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    plt.show()
