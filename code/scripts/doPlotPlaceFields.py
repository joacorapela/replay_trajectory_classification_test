
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/repos/replay_trajectory_classification_test/results/Jaq_03_16_sorted_spike_times_00000000_model.pkl",
                        help="model filename")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/Jaq_03_16_sorted_spike_times_00000000_placeFields.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    model_filename = args.model_filename
    fig_filename_pattern = args.fig_filename_pattern

    with open(model_filename, "rb") as f:
        results = pickle.load(f)
    decoder = results["decoder"]
    linearized_positions = results["linearized_positions"]
    Fs = results["Fs"]

    print("Plotting place fields")
    fig = go.Figure()
    n_positions, n_neurons = decoder.place_fields_.shape
    for n in range(n_neurons):
        trace = go.Scatter(x=decoder.place_fields_.position,
                           y=decoder.place_fields_[:, n],
                           name=f"neuron {n}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Position (cm)")
    fig.update_yaxes(title="Neuron")

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
