
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoding_filename", type=str,
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/repos/replay_trajectory_classification_test/results/Jaq_03_16_sorted_spike_times_00000000_decoding.pkl",
                        help="spikes filename")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/Jaq_03_16_sorted_spike_times_00000000_decoding.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    decoding_filename = args.decoding_filename
    fig_filename_pattern = args.fig_filename_pattern

    with open(decoding_filename, "rb") as f:
        load_res = pickle.load(f)

    decoding_results = load_res["decoding_results"]
    time = load_res["time"]
    linearized_positions = load_res["linearized_positions"]

    time_downsample_factor = 100
    decoding_results = decoding_results.isel(time=slice(0, None, time_downsample_factor))

    fig = go.Figure()

    trace = go.Heatmap(z=decoding_results.acausal_posterior.T,
                       x=decoding_results.acausal_posterior.time,
                       y=decoding_results.acausal_posterior.position,
                       zmin=0.00, zmax=0.05, showscale=False)
    fig.add_trace(trace)

    trace = go.Scatter(x=time, y=linearized_positions.linear_position,
                       mode="markers", marker={"color": "cyan", "size": 5},
                       name="position", showlegend=True)
    fig.add_trace(trace)

    fig.update_xaxes(title="Time (sec)")
    fig.update_yaxes(title="Position (cm)")
    fig.update_coloraxes(showscale=False)

    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
