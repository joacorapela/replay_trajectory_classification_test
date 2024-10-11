
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

import replay_trajectory_classification as rtc
import track_linearization as tl


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/dataFromEric/Jaq_03_16_data/Jaq_03_16_position_info.pkl",
                        help="positions filename")
    parser.add_argument("--spikes_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/dataFromEric/Jaq_03_16_data/Jaq_03_16_sorted_spike_times.pkl",
                        help="spikes filename")
    parser.add_argument("--results_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/repos/replay_trajectory_classification_test/results/Jaq_03_16_sorted_spike_times_model_00000000.pkl",
                        help="results filename")
    args = parser.parse_args()

    positions_filename = args.positions_filename
    spikes_filename = args.spikes_filename
    results_filename = args.results_filename

    positions = pd.read_pickle(positions_filename)
    timestamps = positions.index.to_numpy()
    dt = timestamps[1] - timestamps[0]
    Fs = 1.0 / dt
    spikes_bins = np.append(timestamps-dt/2, timestamps[-1]+dt/2)

    with open(spikes_filename, "rb") as f:
        sorted_spike_times = pickle.load(f)

    binned_spikes_times = np.empty((len(timestamps), len(sorted_spike_times)), dtype=float)
    for n in range(len(sorted_spike_times)):
        binned_spikes_times[:, n] = np.histogram(sorted_spike_times[n], spikes_bins)[0]

    x = positions["nose_x"].to_numpy()
    y = positions["nose_y"].to_numpy()
    positions = np.column_stack((x, y))
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

    linearized_positions = tl.get_linearized_position(positions, track_graph, edge_order=edge_order, edge_spacing=edge_spacing, use_HMM=False)

    place_bin_size = 0.5
    movement_var = 0.25
    environment = rtc.Environment(place_bin_size=place_bin_size,
                                  track_graph=track_graph,
                                  edge_order=edge_order,
                                  edge_spacing=edge_spacing)
    transition_type = rtc.RandomWalk(movement_var=movement_var)

    decoder = rtc.SortedSpikesDecoder(
        environment=environment,
        transition_type=transition_type,
    )

    print("Learning model parameters")
    decoder.fit(linearized_positions.linear_position, binned_spikes_times)

    print(f"Saving model to {results_filename}")
    results = dict(decoder=decoder, linearized_positions=linearized_positions,
                   binned_spikes_times=binned_spikes_times, Fs=Fs)

    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
