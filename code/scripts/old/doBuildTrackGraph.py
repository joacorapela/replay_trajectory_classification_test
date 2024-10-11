
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import replay_trajectory_classification as rtc


def main(argv):
    node_positions = [(120.0, 100.0),
                      (  5.0, 100.0),
                      (  5.0,  55.0),
                      (120.0,  55.0),
                      (  5.0,   8.5),
                      (120.0,   8.5),
                     ]
    edges = [
             (3, 2),
             (0, 1),
             (1, 2),
             (5, 4),
             (4, 2),
            ]
    track_graph = rtc.make_track_graph(node_positions, edges)

    edge_order = [
                  (3, 2),
                  (0, 1),
                  (1, 2),
                  (5, 4),
                  (4, 2),
                 ]
    edge_spacing = [16, 0, 16, 0]


    fig, axs = plt.subplots(nrows=2, ncols=1)
    rtc.plot_track_graph(track_graph, ax=axs[0], draw_edge_labels=True)
    axs[0].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    axs[0].set_xlabel("x-position")
    axs[0].set_ylabel("y-position")
    sns.despine(offset=5)

    rtc.plot_graph_as_1D(track_graph, edge_order, edge_spacing, ax=axs[1])
    fig.show()

    breakpoint()

if __name__ == "__main__":
   main(sys.argv)
