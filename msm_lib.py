# -*- coding: utf-8 -*-


import json
import os
import pandas as pd
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls


################################################################################
# Public functions                                                             #
################################################################################

def load_config(config_path):
    """Load a model configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

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
    log = log[['time', 'station', 'part', 'activity']]
    if pd.api.types.is_numeric_dtype(log['time'].dtype):
        log['time'] = log['time'].map(lambda t: pd.to_datetime(t, unit='s'))
    elif pd.api.types.is_string_dtype(log['time'].dtype):
        log['time'] = log['time'].map(lambda t: pd.to_datetime(t))
    else:
        raise RuntimeError('Unsupported time format')
    log['time'] = log['time'].map(lambda t: t.timestamp())
    log.sort_values(by='time', inplace=True, kind='stable')
    return log


def generate_graph(log, config):
    """Generate a graph model from an event log.

    Args:
        log (pd.DataFrame): Event log.
        config (dict[str, Any]): Model configuration.

    Returns:
        nx.DiGraph: Graph model.
    """
    start_time = time.time()
    graph = nx.DiGraph(name=config['name'])
    part_sublogs, station_sublogs = _extract_sublogs(log)
    traces = _collect_traces(part_sublogs)
    _mine_topology(graph, station_sublogs, traces)
    window = _reconstruct_states(graph, part_sublogs, station_sublogs, log)
    _mine_capacities(graph, log, window)
    _mine_processing_times(graph, part_sublogs, log, window, config)
    _mine_routing_probabilities(graph, part_sublogs, window)
    _mine_transfer_times(graph, part_sublogs, log, window, config)
    _mine_wip_limits(graph, log, window)
    _normalize_data_types(graph)
    graph.graph['generation_time'] = time.time() - start_time
    return graph


def save_graph(graph, config):
    """Save a graph model as a JSON file.

    Args:
        graph (nx.DiGraph): Graph model.
        config (dict[str, Any]): Model configuration.
    """
    path = os.path.join(config['work_path'], config['file_name'])
    data = nx.readwrite.json_graph.adjacency_data(graph)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def load_graph(config):
    """Load a graph model from a JSON file.

    Args:
        config (dict[str, Any]): Model configuration.

    Returns:
        nx.DiGraph: Graph model.
    """
    path = os.path.join(config['work_path'], config['file_name'])
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    graph = nx.readwrite.json_graph.adjacency_graph(data)
    return graph


def show_graph(graph, layout=lambda g: nx.nx_agraph.graphviz_layout(g, prog='circo')):
    """Show a graph model in a figure window.

    Args:
        graph (nx.DiGraph): Graph model.
        layout ((nx.DiGraph) -> dict[Any, np.ndarray]): Layout function.
    """
    name = graph.graph['name']

    labels = dict()
    for station, attributes in graph.nodes.items():
        buffer_capacity = attributes['buffer_capacity']
        machine_capacity = attributes['machine_capacity']
        processing_time = attributes['processing_time']
        labels[station] = f"{station}\n({buffer_capacity}, {machine_capacity}, " \
                          f"{processing_time['mean']:.2f}, {processing_time['std']:.2f})"

    edge_labels = dict()
    for connection, attributes in graph.edges.items():
        routing_probability = attributes['routing_probability']
        transfer_time = attributes['transfer_time']
        edge_labels[connection] = f"({routing_probability:.2f}, " \
                                  f"{transfer_time['mean']:.2f}, {transfer_time['std']:.2f})"

    wip_limits = graph.graph['wip_limits']

    plt.title(name)
    pos = layout(graph)
    colors = list(cls.TABLEAU_COLORS.values())

    subgraphs = list(nx.weakly_connected_components(graph))
    for x in range(len(subgraphs)):
        nx.draw_networkx_nodes(graph, pos, nodelist=subgraphs[x], node_size=50,
                               node_color=colors[x % len(colors)], label=f'{wip_limits[x]}')
    nx.draw_networkx_edges(graph, pos, arrowsize=12)

    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    plt.legend(fontsize=10, title='WIP Limit', title_fontsize=10)
    plt.show(block=True)


################################################################################
# Private functions                                                            #
################################################################################

def _extract_sublogs(log):
    """Extract the sublogs of stations and parts.

    Args:
        log (pd.DataFrame): Event log.

    Returns:
        tuple[dict[Any, pd.DataFrame], dict[Any, pd.DataFrame]]:
            Tuple containing station sublogs and station sublogs.
    """
    parts = log['part'].unique()
    stations = log['station'].unique()
    state = np.full(2 * len(stations), 0)
    log['state'] = [state.copy() for _ in range(log.shape[0])]

    part_sublogs = dict()
    for part in parts:
        sublog = log[log['part'] == part]
        part_sublogs[part] = sublog

    station_sublogs = dict()
    for station in stations:
        sublog = log[log['station'] == station]
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
        if event['activity'] == 'EXIT':
            trace.append(event['station'])
        trace.extend(sublog.loc[sublog['activity'] == 'ENTER', 'station'])
        trace = tuple(trace)
        if trace not in traces.keys():
            traces[trace] = {'parts': set()}
        traces[trace]['parts'].add(part)

    return traces


def _mine_topology(graph, station_sublogs, traces):
    """Mine the topology of the system.

    Args:
        graph (nx.DiGraph): Graph model.
        station_sublogs (dict[Any, pd.DataFrame]): Station sublogs.
        traces (dict[tuple, dict[str, Any]]): Unique traces.
    """
    for station in station_sublogs.keys():
        graph.add_node(station, index=graph.number_of_nodes(), frequency=0,
                       buffer_capacity=0, machine_capacity=0,
                       processing_time={'mean': 0.0, 'std': 0.0})

    for trace in traces.keys():
        for k in range(len(trace) - 1):
            if not graph.has_edge(trace[k], trace[k + 1]):
                graph.add_edge(trace[k], trace[k + 1], frequency=0, routing_probability=0.0,
                               transfer_time={'mean': 0.0, 'std': 0.0})

    graph.graph['start_stations'] = []
    graph.graph['end_stations'] = []
    for station in graph.nodes.keys():
        if graph.in_degree(station) <= 0:
            graph.graph['start_stations'].append(station)
        if graph.out_degree(station) <= 0:
            graph.graph['end_stations'].append(station)

    number_of_subgraphs = nx.number_weakly_connected_components(graph)
    graph.graph['wip_limits'] = [0 for _ in range(number_of_subgraphs)]


def _reconstruct_states(graph, part_sublogs, station_sublogs, log):
    """Reconstruct the state of the system after each event.

    Args:
        graph (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        station_sublogs (dict[Any, pd.DataFrame]): Station sublogs.
        log (pd.DataFrame): Event log.

    Returns:
        list[int]: Definite window.
    """
    start_stations = graph.graph['start_stations']
    end_stations = graph.graph['end_stations']
    window = [0, log.shape[0]]
    for sublog in part_sublogs.values():
        i = sublog.index[-1]
        event = log.loc[i]
        if event['station'] not in end_stations and event['activity'] == 'EXIT':
            if i < window[1]:
                window[1] = i

    log.loc[window[1]:, 'state'] = None
    for sublog in part_sublogs.values():
        sublog.loc[sublog.index >= window[1], 'state'] = None
    for sublog in station_sublogs.values():
        sublog.loc[sublog.index >= window[1], 'state'] = None

    number_of_stations = graph.number_of_nodes()
    previous_state = np.full(2 * number_of_stations, 0)
    floor_state = np.full(2 * number_of_stations, 0)
    for i in range(*window):
        event = log.loc[i]
        state = event['state']
        state[:] = previous_state
        if event['station'] in start_stations and event['activity'] == 'ENTER':
            k_1 = graph.nodes[event['station']]['index']
            k_1_l_2 = 2 * k_1 + 1
            event['state'][k_1_l_2] += 1
        elif event['station'] not in start_stations and event['activity'] == 'ENTER':
            k_1 = graph.nodes[event['station']]['index']
            k_1_l_1 = 2 * k_1
            k_1_l_2 = 2 * k_1 + 1
            event['state'][k_1_l_1] -= 1
            event['state'][k_1_l_2] += 1
            floor_state[k_1_l_1] = min(state[k_1_l_1], floor_state[k_1_l_1])
        elif event['station'] not in end_stations and event['activity'] == 'EXIT':
            sublog = part_sublogs[event['part']]
            k_1 = graph.nodes[event['station']]['index']
            k_2 = graph.nodes[sublog.iloc[sublog.index.get_loc(i) + 1]['station']]['index']
            k_1_l_2 = 2 * k_1 + 1
            k_2_l_1 = 2 * k_2
            state[k_1_l_2] -= 1
            state[k_2_l_1] += 1
            floor_state[k_1_l_2] = min(state[k_1_l_2], floor_state[k_1_l_2])
        elif event['station'] in end_stations and event['activity'] == 'EXIT':
            k_1 = graph.nodes[event['station']]['index']
            k_1_l_2 = 2 * k_1 + 1
            state[k_1_l_2] -= 1
            floor_state[k_1_l_2] = min(state[k_1_l_2], floor_state[k_1_l_2])
        previous_state = state

    for i in range(*window):
        state = log.at[i, 'state']
        np.subtract(state, floor_state, state)

    return window


def _mine_capacities(graph, log, window):
    """Mine the buffer and machine capacities at each station.

    Args:
        graph (nx.DiGraph): Graph model.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
    """
    number_of_stations = graph.number_of_nodes()
    ceiling_state = np.full(2 * number_of_stations, 0)
    for i in range(*window):
        state = log.at[i, 'state']
        np.maximum(state, ceiling_state, ceiling_state)

    for attributes in graph.nodes.values():
        k = attributes['index']
        attributes['buffer_capacity'] = ceiling_state[2 * k]
        attributes['machine_capacity'] = ceiling_state[2 * k + 1]


def _mine_processing_times(graph, part_sublogs, log, window, config):
    """Mine the processing time at each station.

    Args:
        graph (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
        config (dict[str, Any]): Model configuration.
    """
    stations = graph.nodes.keys()
    end_stations = graph.graph['end_stations']
    counts = dict.fromkeys(stations, 0)
    means = dict.fromkeys(stations, 0.0)
    tses = dict.fromkeys(stations, 0.0)
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            event = sublog.iloc[j]
            if i < window[1] and event['activity'] == 'EXIT':
                sample = event['time'] - sublog.iloc[j - 1]['time']
                blocked = False
                station1 = event['station']
                if station1 not in end_stations:
                    station2 = sublog.iloc[j + 1]['station']
                    k_2_l_1 = 2 * graph.nodes[station2]['index']
                    buffer_capacity = graph.nodes[station2]['buffer_capacity']
                    if event['state'][k_2_l_1] >= buffer_capacity:
                        release_delay = config['generation']['release_delay']
                        exit_time = event['time']
                        while i >= 0:
                            i -= 1
                            event = log.loc[i]
                            if exit_time - event['time'] > release_delay:
                                break
                            if event['station'] == station2 and event['activity'] == 'ENTER':
                                blocked = True
                                break
                if not blocked:
                    station = station1
                    counts[station] += 1
                    last_mean = means[station]
                    means[station] = means[station] \
                        + (sample - means[station]) / counts[station]
                    tses[station] = tses[station] \
                        + (sample - last_mean) * (sample - means[station])

    for station, attributes in graph.nodes.items():
        processing_time = dict()
        processing_time['mean'] = means[station]
        if counts[station] <= 1:
            processing_time['std'] = 0.0
        else:
            processing_time['std'] = (tses[station] / (counts[station] - 1)) ** 0.5
        attributes['processing_time'] = processing_time


def _mine_routing_probabilities(graph, part_sublogs, window):
    """Mine the routing probability on each connection.

    Args:
        graph (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        window (list[int]): Definite window.
    """
    end_stations = graph.graph['end_stations']
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            event = sublog.iloc[j]
            if i < window[1]:
                if event['activity'] == 'ENTER':
                    station1 = sublog.iloc[j - 1]['station']
                    station2 = event['station']
                    graph.nodes[station1]['frequency'] += 1
                    graph.edges[station1, station2]['frequency'] += 1
                else:
                    station = event['station']
                    if station in end_stations:
                        graph.nodes[station]['frequency'] += 1

    for station1, station2 in graph.edges.keys():
        connection_frequency = graph.edges[station1, station2]['frequency']
        station_frequency = graph.nodes[station1]['frequency']
        routing_probability = connection_frequency / station_frequency
        graph.edges[station1, station2]['routing_probability'] = routing_probability


def _mine_transfer_times(graph, part_sublogs, log, window, config):
    """Mine the transfer time on each connection.

    Args:
        graph (nx.DiGraph): Graph model.
        part_sublogs (dict[Any, pd.DataFrame]): Part sublogs.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
        config (dict[str, Any]): Model configuration.
    """
    connections = graph.edges.keys()
    counts = dict.fromkeys(connections, 0)
    means = dict.fromkeys(connections, 0.0)
    tses = dict.fromkeys(connections, 0.0)
    for sublog in part_sublogs.values():
        for j in range(1, sublog.shape[0]):
            i = sublog.index[j]
            event = sublog.iloc[j]
            if i < window[1] and event['activity'] == 'ENTER':
                sample = event['time'] - sublog.iloc[j - 1]['time']
                queued = False
                station1 = sublog.iloc[j - 1]['station']
                station2 = event['station']
                k_2_l_2 = 2 * graph.nodes[station2]['index'] + 1
                machine_capacity = graph.nodes[station2]['machine_capacity']
                if event['state'][k_2_l_2] >= machine_capacity:
                    seize_delay = config['generation']['seize_delay']
                    enter_time = event['time']
                    while i >= 0:
                        i -= 1
                        event = log.loc[i]
                        if enter_time - event['time'] > seize_delay:
                            break
                        if event['station'] == station2 and event['activity'] == 'EXIT':
                            queued = True
                            break
                if not queued:
                    connection = (station1, station2)
                    counts[connection] += 1
                    last_mean = means[connection]
                    means[connection] = means[connection] \
                        + (sample - means[connection]) / counts[connection]
                    tses[connection] = tses[connection] \
                        + (sample - last_mean) * (sample - means[connection])

    for connection, attributes in graph.edges.items():
        transfer_time = dict()
        transfer_time['mean'] = means[connection]
        if counts[connection] <= 1:
            transfer_time['std'] = 0.0
        else:
            transfer_time['std'] = (tses[connection] / (counts[connection] - 1)) ** 0.5
        attributes['transfer_time'] = transfer_time


def _mine_wip_limits(graph, log, window):
    """Mine the work-in-progress limit for each subsystem.

    Args:
        graph (nx.DiGraph): Graph model.
        log (pd.DataFrame): Event log.
        window (list[int]): Definite window.
    """
    subgraphs = list(nx.weakly_connected_components(graph))
    state_index_lists = [[] for _ in subgraphs]
    for x in range(len(subgraphs)):
        for station in subgraphs[x]:
            k = graph.nodes[station]['index']
            state_index_lists[x].append(2 * k)
            state_index_lists[x].append(2 * k + 1)

    wip_limits = graph.graph['wip_limits']
    for i in range(*window):
        state = log.at[i, 'state']
        for x in range(len(subgraphs)):
            wip = sum(state[state_index_lists[x]])
            wip_limits[x] = max(wip, wip_limits[x])


def _normalize_data_types(graph):
    """Normalize the data type of each attribute.

    Args:
        graph (nx.DiGraph): Graph model.
    """
    for attributes in graph.nodes.values():
        attributes['buffer_capacity'] = int(attributes['buffer_capacity'])
        attributes['machine_capacity'] = int(attributes['machine_capacity'])
        attributes['processing_time']['mean'] = float(attributes['processing_time']['mean'])
        attributes['processing_time']['std'] = float(attributes['processing_time']['std'])

    for attributes in graph.edges.values():
        attributes['routing_probability'] = float(attributes['routing_probability'])
        attributes['transfer_time']['mean'] = float(attributes['transfer_time']['mean'])
        attributes['transfer_time']['std'] = float(attributes['transfer_time']['std'])

    wip_limits = graph.graph['wip_limits']
    for x in range(len(wip_limits)):
        wip_limits[x] = int(wip_limits[x])
