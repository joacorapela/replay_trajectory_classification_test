
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import replay_trajectory_classification as rtc
import track_linearization as tl


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoding_start_secs", type=int,
                        help="first time to decode (secs)", default=0)
    parser.add_argument("--decoding_duration_secs", type=int,
                        help="duration of segment to decode (sec)", default=100)
    parser.add_argument("--positions_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/dataFromEric/Jaq_03_16_data/Jaq_03_16_position_info.pkl",
                        help="positions filename")
    parser.add_argument("--spikes_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/dataFromEric/Jaq_03_16_data/Jaq_03_16_sorted_spike_times.pkl",
                        help="spikes filename")
    parser.add_argument("--model_filename", type=str,
                        default="../../results/Jaq_03_16_sorted_spike_times_model_00000000.pkl",
                        help="spikes filename")
    parser.add_argument("--decodings_filename", type=str,
                        default="../../results/Jaq_03_16_sorted_spike_times_decodings_00000000.pkl",
                        help="spikes filename")
    args = parser.parse_args()

    decoding_start_secs = args.decoding_start_secs 
    decoding_duration_secs = args.decoding_duration_secs 
    model_filename = args.model_filename
    decodings_filename = args.decodings_filename

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

    environment = rtc.Environment(place_bin_size=0.5,
                                  track_graph=track_graph,
                                  edge_order=edge_order,
                                  edge_spacing=edge_spacing)
    transition_type = rtc.RandomWalk(movement_var=0.25)

    decoder = rtc.SortedSpikesDecoder(
        environment=environment,
        transition_type=transition_type,
    )

    print("Learning model parameters")
    decoder.fit(linearized_positions.linear_position, binned_spikes_times)

    print(f"Saving model to {model_filename}")
    results = dict(decoder=decoder, linearized_positions=linearized_positions, Fs=Fs)
    with open(model_filename, "wb") as f:
        pickle.dump(results, f)

    print("Plotting place fields")
    fig, ax = plt.subplots(figsize=(10, 3))
    (decoder.place_fields_ * Fs).plot(x="position", hue="neuron", add_legend=False, ax=ax)
    ax.set_xlabel('Linear Position')
    ax.set_ylabel('Firing Rate')
    ax.set_xlim((0, linearized_positions.linear_position.max()))
    sns.despine()

    with open(model_filename, "rb") as f:
        model_results = pickle.load(f)

    print("Decoding positions from spikes")
    time_ind = slice(decoding_start_samples,
                     decoding_start_samples + decoding_duration_samples)
    time = np.arange(model_results["linearized_positions"].linear_position.size) / model_results["Fs"]
    decoding_results = decoder.predict(model_results["binned_spikes_times"][time_ind], time=time[time_ind])

    print("Plotting decoding results")
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True, constrained_layout=True)
    spike_time_ind, neuron_ind = np.nonzero(binned_spikes_times[time_ind])
    cmap = plt.get_cmap('tab20')
    # c1 = [cmap.colors[ind] for ind in neuron_ind]
    # axes[0].scatter(time[time_ind][spike_time_ind], neuron_ind, clip_on=False, s=1, c=c1)
    axes[0].scatter(time[time_ind][spike_time_ind], neuron_ind, clip_on=False, s=1)
    axes[0].set_ylabel('Cells')

    decoding_results.acausal_posterior.plot(x="time", y="position", ax=axes[1],
                                   robust=True, cmap="bone_r",
                                   vmin=0.0, vmax=0.05, clip_on=False)
    axes[1].scatter(time[time_ind], linearized_positions.iloc[time_ind].linear_position, color="magenta", s=1, clip_on=False)
    axes[1].set_ylabel('Linear Position')
    axes[1].set_xlabel('Time')
    sns.despine(offset=5)

    breakpoint()

if __name__ == "__main__":
   main(sys.argv)
