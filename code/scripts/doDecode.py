
import sys
import argparse
import pickle
import numpy as np
import pandas as pd


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoding_start_secs", type=int,
                        help="first time to decode (secs)", default=0)
    parser.add_argument("--decoding_duration_secs", type=int,
                        help="duration of segment to decode (sec)", default=100)
    parser.add_argument("--model_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/repos/replay_trajectory_classification_test/results/Jaq_03_16_sorted_spike_times_00000000_model.pkl",
                        help="spikes filename")
    parser.add_argument("--decoding_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/repos/replay_trajectory_classification_test/results/Jaq_03_16_sorted_spike_times_00000000_decoding.pkl",
                        help="spikes filename")
    args = parser.parse_args()

    decoding_start_secs = args.decoding_start_secs
    decoding_duration_secs = args.decoding_duration_secs
    model_filename = args.model_filename
    decoding_filename = args.decoding_filename

    with open(model_filename, "rb") as f:
        model_results = pickle.load(f)

    decoder = model_results["decoder"]
    Fs = model_results["Fs"]
    binned_spikes_times = model_results["binned_spikes_times"]
    linearized_positions = model_results["linearized_positions"]

    print("Decoding positions from spikes")
    decoding_start_samples = int(decoding_start_secs * Fs)
    decoding_duration_samples = int(decoding_duration_secs * Fs)
    time_ind = slice(decoding_start_samples, decoding_start_samples + decoding_duration_samples)
    time = np.arange(linearized_positions.linear_position.size) / Fs
    decoding_results = decoder.predict(binned_spikes_times[time_ind], time=time[time_ind])

    results = dict(decoding_results=decoding_results, time=time[time_ind],
                   linearized_positions=linearized_positions.iloc[time_ind],
                   binned_spikes_times=binned_spikes_times[time_ind])

    print(f"Saving decoding results to {decoding_filename}")
    with open(decoding_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()

if __name__ == "__main__":
   main(sys.argv)
