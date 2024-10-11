
import sys
import argparse
import pickle
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_data(folder_path):
    folder_path = pathlib.Path(folder_path)
    position_info = pd.read_pickle(folder_path / "Jaq_03_16_position_info.pkl")

    with open(folder_path / "Jaq_03_16_sorted_spike_times.pkl", "rb") as f:
        sorted_spike_times = pickle.load(f)

    with open(folder_path / "Jaq_03_16_clusterless_spike_times.pkl", "rb") as f:
        clusterless_spike_times = pickle.load(f)
    with open(
        folder_path / "Jaq_03_16_clusterless_spike_waveform_features.pkl", "rb"
    ) as f:
        clusterless_spike_waveform_features = pickle.load(f)

    position_time = np.asarray(position_info.index).astype(float)
    position1D = np.asarray(position_info.linear_position).astype(float)
    position2D = np.asarray(position_info[["nose_x", "nose_y"]]).astype(float)

    return (
        position_time,
        position1D,
        position2D,
        sorted_spike_times,
        clusterless_spike_times,
        clusterless_spike_waveform_features,
    )


def getSpikesTimesPlotOneTrial(spikes_times, title="",
                               xlabel="Time (sec)", ylabel="Neuron",
                               event_line_color="rgba(0, 0, 255, 0.2)",
                               event_line_width=5):
    nNeurons = len(spikes_times)
    min_time = np.inf
    max_time = -np.inf
    fig = go.Figure()
    for n in range(nNeurons):
        # workaround because if a trial contains only one spike spikes_times[n]
        # does not respond to the len function
        if len(spikes_times[n].shape) == 0:
            x = [spikes_times[n]]
        else:
            x = spikes_times[n]
        if len(x) > 0:
            min_time = min(min_time, x.min())
            max_time = max(max_time, x.max())
        trace = go.Scatter(
            x=x,
            y=n*np.ones(len(x)),
            mode="markers",
            marker=dict(size=3, color="black"),
            showlegend=False,
            # hoverinfo="skip",
        )
        fig.add_trace(trace)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title)
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    return fig


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="data filename",
                        default="/nfs/gatsbystor/rapela/bbsrc23Project/dataFromEric/Jaq_03_16_data")
    parser.add_argument("--from_neuron", type=int, help="first neuron to plot",
                        default=50)
    parser.add_argument("--to_neuron", type=int, help="last neuron to plot",
                        default=80)
    args = parser.parse_args()

    data_folder = args.data_folder
    from_neuron = args.from_neuron
    to_neuron = args.to_neuron

    (
        position_time,
        position1D,
        position2D,
        sorted_spike_times,
        clusterless_spike_times,
        clusterless_spike_waveform_features,
    ) = load_data(data_folder)

    breakpoint()
#     fig = go.Figure()
#     trace = go.Scatter(x=position2D[:, 0], y=position2D[:, 1],
#                        text=[f"Time: {t}" for t in position_time],
#                        hovertemplate="X: %{x}<br>" +
#                                      "Y: %{y}<br>" +
#                                      "%{text}")
#     fig.add_trace(trace)
#     fig.update_xaxes(title="X")
#     fig.update_yaxes(title="Y")
#     fig.write_html("../../figures/pos2D.html")
# 
#     fig.show()
# 
#     fig = go.Figure()
#     trace = go.Scatter(x=position_time, y=position1D)
#     fig.add_trace(trace)
#     fig.update_xaxes(title="Time (sec)")
#     fig.update_yaxes(title="Position 1D")
#     fig.write_html("../../figures/pos1D.html")
# 
#     fig.show()
# 
#     fig = getSpikesTimesPlotOneTrial(spikes_times=sorted_spike_times)
#     fig.update_yaxes(title="Spikes Times")
#     fig.write_html("../../figures/spikesTimes.html")
# 
#     fig.show()
# 
    pos1D_with_spike = [[] for i in range(len(sorted_spike_times))]
    for i in range(len(sorted_spike_times)):
        indices = np.array([np.abs(position_time - sorted_spike_times[i][j]).argmin()
                            for j in range(len(sorted_spike_times[i]))])
        pos1D_with_spike[i] = position1D[indices]

    fig = getSpikesTimesPlotOneTrial(spikes_times=pos1D_with_spike,
                                     xlabel="Linear Position")
    fig.update_yaxes(title="Positions with Spikes")
    fig.write_html("../../figures/pos1DForpikes.html")

    fig.show()

    pos2D_with_spike = [[] for i in range(len(sorted_spike_times))]
    for i in range(from_neuron, to_neuron):
        indices = np.array([np.abs(position_time - sorted_spike_times[i][j]).argmin()
                            for j in range(len(sorted_spike_times[i]))])
        pos2D_with_spike[i] = position2D[indices, :]

    fig = go.Figure()
    # for i in range(len(sorted_spike_times)):
    for i in range(from_neuron, to_neuron):
        trace = go.Scatter(x=pos2D_with_spike[i][:, 0],
                           y=pos2D_with_spike[i][:, 1],
                           mode="markers", name=f"neuron {i}")
        fig.add_trace(trace)
    fig.update_xaxes(title="X")
    fig.update_yaxes(title="Y")
    fig.write_html(f"../../figures/pos2DForSpikesFrom{from_neuron}To{to_neuron}.html")

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
