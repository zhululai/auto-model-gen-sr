# -*- coding: utf-8 -*-


import json
import os
import pandas as pd
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.options.mode.copy_on_write = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


################################################################################
# Public functions                                                             #
################################################################################

def load_config(config_path):
    """Load a model configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON file.

    Returns:
        dict[str, Any]: Model configuration.
    """
    with open(config_path, encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config


def load_log(config):
    """Load an event log from a CSV file.

    Args:
        config (dict[str, Any]): Model configuration.

    Returns:
        pd.DataFrame: Event log.
    """
    path = os.path.join(config['work_path'], config['log']['path'])
    log = pd.read_csv(path)
    log.rename(columns=config['log']['mappings'], inplace=True)
    columns = ['time', 'station', 'part', 'activity']
    columns = [column for column in log.columns if column not in columns]
    log.drop(columns=columns, inplace=True)
    if pd.api.types.is_numeric_dtype(log['time'].dtype):
        log['time'] = log['time'].map(lambda t: pd.to_datetime(t, unit='s'))
    elif pd.api.types.is_string_dtype(log['time'].dtype):
        log['time'] = log['time'].map(lambda t: pd.to_datetime(t))
    else:
        raise RuntimeError("Unsupported time format")
    log['time'] = log['time'].map(lambda t: t.timestamp())
    log.sort_values(by='time', inplace=True, kind='stable')
    return log


def generate_model(log, config):
    """Generate a graph model from an event log.

    Args:
        log (pd.DataFrame): Event log.
        config (dict[str, Any]): Model configuration.

    Returns:
        nx.DiGraph: Graph model.
    """
    start_time = time.time()
    model = nx.DiGraph(name=config['name'])
    part_sublogs, station_sublogs = _extract_sublogs(log)
    traces = _collect_traces(part_sublogs)
    _mine_topology(model, traces, station_sublogs)
    window = [-1, log.shape[0]]
    _reconstruct_states(model, part_sublogs, station_sublogs, log, window)
    _mine_capacities(model, log, window)
    _mine_processing_times(model, part_sublogs, log, window, config)
    _mine_routing_probabilities(model, part_sublogs, window)
    _mine_transfer_times(model, part_sublogs, log, window, config)
    _mine_wip_limits(model, log, window)
    _standardize_data_types(model)
    model.graph['generation_time'] = time.time() - start_time
    return model


def save_model(model, config):
    """Save a graph model as a JSON file.

    Args:
        model (nx.DiGraph): Graph model.
        config (dict[str, Any]): Model configuration.
    """
    path = os.path.join(config['work_path'], config['file_name'])
    data = nx.readwrite.json_graph.adjacency_data(model)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def load_model(config):
    """Load a graph model from a JSON file.

    Args:
        config (dict[str, Any]): Model configuration.

    Returns:
        nx.DiGraph: Graph model.
    """
    path = os.path.join(config['work_path'], config['file_name'])
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    model = nx.readwrite.json_graph.adjacency_graph(data)
    return model


def show_model(model, layout=lambda g: nx.nx_agraph.graphviz_layout(g, prog='circo')):
    """Show a graph model in a figure window.

    Args:
        model (nx.DiGraph): Graph model.
        layout ((nx.DiGraph) -> dict[Any, np.ndarray]): Layout function.
    """
    name = model.graph['name']
    wip_limits = model.graph['wip_limits']
    submodels = list(nx.weakly_connected_components(model))

    plt.title(name, fontsize=10)
    pos = layout(model)
    cmap = plt.get_cmap('gist_rainbow')
    black = mpl.colors.to_rgba_array('black')
    colors = cmap(np.linspace(0.0, 1.0, len(submodels)))
    colors = [colors[x].reshape(1, -1) for x in range(len(colors))]

    stations = list(model.nodes.keys())
    paths = []
    for station in stations:
        for x in range(len(submodels)):
            if station in submodels[x]:
                path = nx.draw_networkx_nodes(model, pos, nodelist=[station],
                                              node_color=colors[x], edgecolors=black)
                paths.append(path)
                break

    connections = list(model.edges.keys())
    patches = []
    for connection in connections:
        for x in range(len(submodels)):
            if connection[0] in submodels[x]:
                patch = nx.draw_networkx_edges(model, pos, edgelist=[connection],
                                               edge_color=colors[x], arrowsize=20)
                patch[0].set_edgecolor(black)
                patches.append(patch[0])
                break
    nx.draw_networkx_labels(model, pos, font_size=8)

    handles = []
    labels = []
    for x in range(len(submodels)):
        handles.append(mpl.patches.Rectangle((0, 0), 0, 0, color=colors[x]))
        labels.append(f"{wip_limits[x]}")
    plt.legend(handles, labels, fontsize=8, title='WIP Limit', title_fontsize=8)

    annotation = plt.annotate(None, xy=(0.0, 0.0), xytext=(0.0, 0.0),
                              textcoords='offset points',
                              arrowprops={'arrowstyle': '-'},
                              multialignment='left',
                              bbox={'boxstyle': 'round', 'facecolor': 'white'},
                              fontsize=8, visible=False, zorder=6)

    def handle_mouse_motion(event):
        """Handle the mouse motion in a figure window.

        Args:
            event (mpl.backend_bases.Event): Mouse motion event.
        """
        is_inside = False
        text = ""
        for x in range(len(paths)):
            if is_inside:
                break
            is_inside, _ = paths[x].contains(event)
            if is_inside:
                station_ = stations[x]
                attributes = model.nodes[station_]
                annotation.xy = pos[station_]
                text += "Station: " + str(station_) + "\n"
                text += "Buffer Capacity: "
                text += get_display_text(attributes['buffer_capacity'], 1) + "\n"
                text += "Machine Capacity: "
                text += get_display_text(attributes['machine_capacity'], 1) + "\n"
                text += "Processing Time: "
                text += get_display_text(attributes['processing_time'], 1)
        for x in range(len(patches)):
            if is_inside:
                break
            is_inside, _ = patches[x].contains(event, radius=5)
            if is_inside:
                connection_ = connections[x]
                attributes = model.edges[connection_]
                origin_xy = pos[connection_[0]]
                destination_xy = pos[connection_[1]]
                annotation.xy = ((origin_xy[0] + destination_xy[0]) / 2,
                                 (origin_xy[1] + destination_xy[1]) / 2)
                text += "Origin Station: " + str(connection_[0]) + "\n"
                text += "Destination Station: " + str(connection_[1]) + "\n"
                text += "Routing Probability: "
                text += get_display_text(attributes['routing_probability'], 1) + "\n"
                text += "Transfer Time: "
                text += get_display_text(attributes['transfer_time'], 1)
        if is_inside:
            point_xy = plt.gca().transData.transform(annotation.xy)
            center_xy = plt.gcf().transFigure.transform((0.5, 0.5))
            if point_xy[0] < center_xy[0]:
                annotation.set(horizontalalignment='left')
            else:
                annotation.set(horizontalalignment='right')
            if point_xy[1] < center_xy[1]:
                annotation.xyann = (0.0, 30.0)
                annotation.set(verticalalignment='bottom')
            else:
                annotation.xyann = (0.0, -30.0)
                annotation.set(verticalalignment='top')
            annotation.set_text(text)
            annotation.set_visible(True)
        else:
            if annotation.get_visible():
                annotation.set_visible(False)

    display_names = {'mean': 'Mean', 'std': "Standard Deviation"}

    def get_display_text(value, level):
        """Get the display text of an attribute value.

        Args:
            value (Any): Attribute value.
            level (int): Indent level.

        Return:
            text (str): Display text.
        """
        if isinstance(value, list):
            text = ""
            for x in range(len(value)):
                x += 1
                y = value[x - 1]
                text += "\n" + "     " * level
                text += str(x) + ": " + get_display_text(y, level + 1)
        elif isinstance(value, dict):
            text = ""
            for x, y in value.items():
                if x in display_names.keys():
                    x = display_names[x]
                text += "\n" + "     " * level
                text += str(x) + ": " + get_display_text(y, level + 1)
        elif isinstance(value, float) or isinstance(value, np.floating):
            text = f"{value:.2f}"
        else:
            text = str(value)
        return text

    plt.ion()
    plt.connect("motion_notify_event", handle_mouse_motion)
    plt.show(block=True)


################################################################################
# Private functions                                                            #
################################################################################

def _extract_sublogs(log):
    """Extract the sublogs of stations and parts.

    Args:
        log (pd.DataFrame): Event log.

    Returns:
        tuple[dict[Any, pd.DataFrame], dict[Any, pd.DataFrame]]: Tuple
            containing part sublogs and station sublogs.
    """
    parts = log['part'].unique()
    part_sublogs = dict()
    for part in parts:
        sublog = log[log['part'] == part].copy()
        part_sublogs[part] = sublog

    stations = log['station'].unique()
    station_sublogs = dict()
    for station in stations:
        sublog = log[log['station'] == station].copy()
        station_sublogs[station] = sublog

    return part_sublogs, station_sublogs


def _collect_traces(part_sublogs):
    """Collect the unique traces of parts.

    Args:
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.

    Returns:
        dict[tuple, dict[str, Any]]: Unique traces.
    """
    traces = dict()
    for part, sublog in part_sublogs.items():
        trace = []
        event = sublog.iloc[0]
        station = event['station']
        activity = event['activity']
        if activity == 'EXIT':
            trace.append(station)
        trace.extend(sublog.loc[sublog['activity'] == 'ENTER', 'station'])
        trace = tuple(trace)
        if trace not in traces.keys():
            traces[trace] = {'parts': set()}
        traces[trace]['parts'].add(part)

    return traces


def _mine_topology(model, traces, station_sublogs):
    """Mine the topology of the system.

    Args:
        model (nx.DiGraph): Graph model.
        traces (dict[tuple, dict[str, Any]]): Unique traces.
        station_sublogs (dict[Any, pd.DataFrame]): Station sublogs.
    """
    for station in station_sublogs.keys():
        model.add_node(station, index=model.number_of_nodes(), is_source=False,
                       is_sink=False, buffer_capacity=0, machine_capacity=0,
                       processing_time={'mean': 0.0, 'std': 0.0})

    for trace in traces.keys():
        for k in range(len(trace) - 1):
            if not model.has_edge(trace[k], trace[k + 1]):
                model.add_edge(trace[k], trace[k + 1], routing_probability=0.0,
                               transfer_time={'mean': 0.0, 'std': 0.0})

    for station in model.nodes.keys():
        model.nodes[station]['is_source'] = (model.in_degree(station) <= 0)
        model.nodes[station]['is_sink'] = (model.out_degree(station) <= 0)

    submodels = list(nx.weakly_connected_components(model))
    model.graph['wip_limits'] = [0 for _ in range(len(submodels))]


def _reconstruct_states(model, part_sublogs, station_sublogs, log, window):
    """Reconstruct the system state after each event.

    Args:
        model (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        station_sublogs (dict[Any, pd.DataFrame]): Station sublogs.
        log (pd.DataFrame): Event log.
        window list[int]: Definite window.
    """
    for sublog in part_sublogs.values():
        i = sublog.index[-1]
        event = log.loc[i]
        station = event['station']
        activity = event['activity']
        if not model.nodes[station]['is_sink'] and activity == 'EXIT':
            if i < window[1]:
                window[1] = i

    stations = model.nodes.keys()
    log['state'] = None
    for i in range(window[0] + 1, window[1]):
        log.at[i, 'state'] = np.full(2 * len(stations), 0)
    for sublog in part_sublogs.values():
        sublog['state'] = None
        sublog.update(log['state'])
    for sublog in station_sublogs.values():
        sublog['state'] = None
        sublog.update(log['state'])

    previous_state = np.full(2 * len(stations), 0)
    floor_state = np.full(2 * len(stations), 0)
    for i in range(window[0] + 1, window[1]):
        event = log.loc[i]
        state = event['state']
        state[:] = previous_state
        station = event['station']
        part = event['part']
        activity = event['activity']
        if model.nodes[station]['is_source'] and activity == 'ENTER':
            k_1 = model.nodes[station]['index']
            k_1_l_2 = 2 * k_1 + 1
            state[k_1_l_2] += 1
        elif not model.nodes[station]['is_source'] and activity == 'ENTER':
            k_1 = model.nodes[station]['index']
            k_1_l_1 = 2 * k_1
            k_1_l_2 = 2 * k_1 + 1
            state[k_1_l_1] -= 1
            state[k_1_l_2] += 1
            floor_state[k_1_l_1] = min(state[k_1_l_1], floor_state[k_1_l_1])
        elif not model.nodes[station]['is_sink'] and activity == 'EXIT':
            sublog = part_sublogs[part]
            k_1 = model.nodes[station]['index']
            k_2 = model.nodes[sublog.iloc[sublog.index.get_loc(i) + 1]['station']]['index']
            k_1_l_2 = 2 * k_1 + 1
            k_2_l_1 = 2 * k_2
            state[k_1_l_2] -= 1
            state[k_2_l_1] += 1
            floor_state[k_1_l_2] = min(state[k_1_l_2], floor_state[k_1_l_2])
        elif model.nodes[station]['is_sink'] and activity == 'EXIT':
            k_1 = model.nodes[station]['index']
            k_1_l_2 = 2 * k_1 + 1
            state[k_1_l_2] -= 1
            floor_state[k_1_l_2] = min(state[k_1_l_2], floor_state[k_1_l_2])
        previous_state = state

    for i in range(window[0] + 1, window[1]):
        state = log.at[i, 'state']
        np.subtract(state, floor_state, state)

    return window


def _mine_capacities(model, log, window):
    """Mine the buffer and machine capacities at each station.

    Args:
        model (nx.DiGraph): Graph model.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
    """
    stations = model.nodes.keys()
    ceiling_state = np.full(2 * len(stations), 0)
    for i in range(window[0] + 1, window[1]):
        state = log.at[i, 'state']
        np.maximum(state, ceiling_state, ceiling_state)

    for attributes in model.nodes.values():
        k = attributes['index']
        attributes['buffer_capacity'] = ceiling_state[2 * k]
        attributes['machine_capacity'] = ceiling_state[2 * k + 1]


def _mine_processing_times(model, part_sublogs, log, window, config):
    """Mine the processing time at each station.

    Args:
        model (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
        config (dict[str, Any]): Model configuration.
    """
    stations = model.nodes.keys()
    counts = dict.fromkeys(stations, 0)
    means = dict.fromkeys(stations, 0.0)
    tses = dict.fromkeys(stations, 0.0)
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue
            event = sublog.iloc[j]
            activity = event['activity']
            if activity == 'ENTER':
                continue
            exit_event = event
            enter_event = sublog.iloc[j - 1]
            sample = exit_event['time'] - enter_event['time']
            is_blocked = False
            station = event['station']
            if not model.nodes[station]['is_sink']:
                next_station = sublog.iloc[j + 1]['station']
                k_2_l_1 = 2 * model.nodes[next_station]['index']
                buffer_state = event['state'][k_2_l_1]
                buffer_capacity = model.nodes[next_station]['buffer_capacity']
                if buffer_state >= buffer_capacity:
                    release_delay = config['generation']['release_delay']
                    event = exit_event
                    while i - 1 > window[0] \
                            and exit_event['time'] - event['time'] <= release_delay:
                        i -= 1
                        event = log.loc[i]
                        buffer_state = event['state'][k_2_l_1]
                        if buffer_state >= buffer_capacity:
                            is_blocked = True
                            break
            if not is_blocked:
                counts[station] += 1
                last_mean = means[station]
                means[station] = means[station] \
                    + (sample - means[station]) / counts[station]
                tses[station] = tses[station] \
                    + (sample - last_mean) * (sample - means[station])

    for station, attributes in model.nodes.items():
        processing_time = attributes['processing_time']
        processing_time['mean'] = means[station]
        if counts[station] > 1:
            processing_time['std'] = (tses[station] / (counts[station] - 1)) ** 0.5


def _mine_routing_probabilities(model, part_sublogs, window):
    """Mine the routing probability on each connection.

    Args:
        model (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        window (list[int]): Definite window.
    """
    connection_frequencies = dict()
    station_frequencies = dict()
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue
            event = sublog.iloc[j]
            activity = event['activity']
            if activity == 'EXIT':
                continue
            station = event['station']
            previous_station = sublog.iloc[j - 1]['station']
            connection = (previous_station, station)
            if connection not in connection_frequencies.keys():
                connection_frequencies[connection] = 0
            connection_frequencies[connection] += 1
            if previous_station not in station_frequencies.keys():
                station_frequencies[previous_station] = 0
            station_frequencies[previous_station] += 1

    for connection in model.edges.keys():
        connection_frequency = connection_frequencies[connection]
        station_frequency = station_frequencies[connection[0]]
        routing_probability = connection_frequency / station_frequency
        model.edges[connection]['routing_probability'] = routing_probability


def _mine_transfer_times(model, part_sublogs, log, window, config):
    """Mine the transfer time on each connection.

    Args:
        model (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
        config (dict[str, Any]): Model configuration.
    """
    connections = model.edges.keys()
    counts = dict.fromkeys(connections, 0)
    means = dict.fromkeys(connections, 0.0)
    tses = dict.fromkeys(connections, 0.0)
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            if i <= window[0] or i >= window[1]:
                continue
            event = sublog.iloc[j]
            activity = event['activity']
            if activity == 'EXIT':
                continue
            enter_event = event
            exit_event = sublog.iloc[j - 1]
            sample = enter_event['time'] - exit_event['time']
            is_queued = False
            station = enter_event['station']
            previous_station = exit_event['station']
            k_2_l_2 = 2 * model.nodes[station]['index'] + 1
            machine_state = event['state'][k_2_l_2]
            machine_capacity = model.nodes[station]['machine_capacity']
            if machine_state >= machine_capacity:
                seize_delay = config['generation']['seize_delay']
                event = enter_event
                while i - 1 > window[0] \
                        and enter_event['time'] - event['time'] <= seize_delay:
                    i -= 1
                    event = log.loc[i]
                    machine_state = event['state'][k_2_l_2]
                    if machine_state >= machine_capacity:
                        is_queued = True
                        break
            if not is_queued:
                connection = (previous_station, station)
                counts[connection] += 1
                last_mean = means[connection]
                means[connection] = means[connection] \
                    + (sample - means[connection]) / counts[connection]
                tses[connection] = tses[connection] \
                    + (sample - last_mean) * (sample - means[connection])

    for connection, attributes in model.edges.items():
        transfer_time = attributes['transfer_time']
        transfer_time['mean'] = means[connection]
        if counts[connection] > 1:
            transfer_time['std'] = (tses[connection] / (counts[connection] - 1)) ** 0.5


def _mine_wip_limits(model, log, window):
    """Mine the work-in-progress limit for each subsystem.

    Args:
        model (nx.DiGraph): Graph model.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
    """
    submodels = list(nx.weakly_connected_components(model))
    state_index_lists = [[] for _ in submodels]
    for x in range(len(submodels)):
        for station in submodels[x]:
            k = model.nodes[station]['index']
            state_index_lists[x].append(2 * k)
            state_index_lists[x].append(2 * k + 1)

    wip_limits = model.graph['wip_limits']
    for i in range(window[0] + 1, window[1]):
        state = log.at[i, 'state']
        for x in range(len(submodels)):
            wip = sum(state[state_index_lists[x]])
            wip_limits[x] = max(wip, wip_limits[x])


def _standardize_data_types(model):
    """Standardize the data type of each attribute.

    Args:
        model (nx.DiGraph): Graph model.
    """
    for attributes in model.nodes.values():
        attributes['buffer_capacity'] = int(attributes['buffer_capacity'])
        attributes['machine_capacity'] = int(attributes['machine_capacity'])
        attributes['processing_time']['mean'] = float(attributes['processing_time']['mean'])
        attributes['processing_time']['std'] = float(attributes['processing_time']['std'])

    for attributes in model.edges.values():
        attributes['routing_probability'] = float(attributes['routing_probability'])
        attributes['transfer_time']['mean'] = float(attributes['transfer_time']['mean'])
        attributes['transfer_time']['std'] = float(attributes['transfer_time']['std'])

    wip_limits = model.graph['wip_limits']
    for x in range(len(wip_limits)):
        wip_limits[x] = int(wip_limits[x])
