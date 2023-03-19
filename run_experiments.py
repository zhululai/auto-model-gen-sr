# -*- coding: utf-8 -*-


import os
import msm_lib as msm
import matplotlib.pyplot as plt
import math


SYSTEM_INSTANCES = ['serial-arena', 'parallel-arena', 'cyclic-arena']
WIP_LIMITS = [6, 9, 12]
LOG_LENGTHS = [100, 1000, 10000]
NUMBER_OF_REPLICAS = 5
FIGURES_FOLDER = 'figures'


# Generate a graph model for each event log
# for instance in SYSTEM_INSTANCES:
#     for limit in WIP_LIMITS:
#         for length in LOG_LENGTHS:
#             for r in range(NUMBER_OF_REPLICAS):
#                 r += 1
#                 log_id = 'wip' + str(limit) + '-log' + str(length) + '-' + str(r)
#                 config_path = os.path.join(instance, log_id, 'config.json')
#                 config = msm.load_config(config_path)
#                 log = msm.load_log(config)
#                 graph = msm.generate_graph(log, config)
#                 msm.save_graph(graph, config)


# Calculate the average generation times of graph models
average_times = dict()
for instance in SYSTEM_INSTANCES:
    average_times[instance] = dict()
    for length in LOG_LENGTHS:
        average_times[instance][length] = 0.0
        for limit in WIP_LIMITS:
            for r in range(NUMBER_OF_REPLICAS):
                r += 1
                log_id = 'wip' + str(limit) + '-log' + str(length) + '-' + str(r)
                config_path = os.path.join(instance, log_id, 'config.json')
                config = msm.load_config(config_path)
                graph = msm.load_graph(config)
                average_times[instance][length] += graph.graph['generation_time']
        average_times[instance][length] /= len(WIP_LIMITS) * NUMBER_OF_REPLICAS


# Draw the average generation times of graph models
plt.rcParams.update({'font.size': 16})
x_ticks = [(i + 1) for i in range(len(SYSTEM_INSTANCES))]
plt.xticks(x_ticks, [instance[:instance.find('-')].title() for instance in SYSTEM_INSTANCES])
plt.xlabel('System Instance')
plt.ylim([0.0, 120.0])
plt.yticks([10.0 * q for q in range(13)])
plt.ylabel('Average Generation Time (s)')
plt.grid(True, axis='y', linestyle=(0, (1, 10)))
width = 1 / (len(LOG_LENGTHS) + 1)
x = [p - (len(LOG_LENGTHS) - 1) * (width / 2) for p in x_ticks]
for length in LOG_LENGTHS:
    plt.bar(x, [average_times[instance][length] for instance in SYSTEM_INSTANCES], width)
    x = [p + width for p in x]
plt.legend([r'$10^{' + str(round(math.log10(length))) + '}$' for length in LOG_LENGTHS], title='Log Length', ncols=3)
figure_name = 'generation_times.eps'
figure_path = os.path.join(FIGURES_FOLDER, figure_name)
plt.savefig(figure_path, format='eps', bbox_inches='tight')
plt.show(block=True)


def calculate_error(estimate, reference):
    """Calculate the absolute relative error of an estimate.

    Args:
        estimate (int | float): Estimate.
        reference (int | float): Reference.

    Returns:
        float: Absolute relative error.
    """
    if reference <= 0:
        if estimate <= 0:
            error = 0.0
        else:
            error = float('inf')
    else:
        error = abs((estimate - reference) / reference)
    return error


# Calculate the average maximum errors of attribute estimates
average_errors = dict()
for instance in SYSTEM_INSTANCES:
    average_errors[instance] = dict()
    config_path = os.path.join(instance, 'config.json')
    config = msm.load_config(config_path)
    config['file_name'] = 'truth.json'
    truth = msm.load_graph(config)
    for limit in WIP_LIMITS:
        average_errors[instance][limit] = dict()
        for length in LOG_LENGTHS:
            average_errors[instance][limit][length] = dict()
            average_errors[instance][limit][length]['buffer_capacity'] = 0.0
            average_errors[instance][limit][length]['machine_capacity'] = 0.0
            average_errors[instance][limit][length]['processing_time'] = dict()
            average_errors[instance][limit][length]['processing_time']['mean'] = 0.0
            average_errors[instance][limit][length]['processing_time']['std'] = 0.0
            average_errors[instance][limit][length]['routing_probability'] = 0.0
            average_errors[instance][limit][length]['transfer_time'] = dict()
            average_errors[instance][limit][length]['transfer_time']['mean'] = 0.0
            average_errors[instance][limit][length]['transfer_time']['std'] = 0.0
            average_errors[instance][limit][length]['wip_limit'] = 0.0
            for r in range(NUMBER_OF_REPLICAS):
                r += 1
                log_id = 'wip' + str(limit) + '-log' + str(length) + '-' + str(r)
                config_path = os.path.join(instance, log_id, 'config.json')
                config = msm.load_config(config_path)
                graph = msm.load_graph(config)

                errors = dict()
                errors['buffer_capacity'] = 0.0
                errors['machine_capacity'] = 0.0
                errors['processing_time'] = dict()
                errors['processing_time']['mean'] = 0.0
                errors['processing_time']['std'] = 0.0
                stations = graph.nodes.keys()
                for station in stations:
                    buffer_capacity_error = calculate_error(graph.nodes[station]['buffer_capacity'],
                                                            truth.nodes[station]['buffer_capacity'])
                    errors['buffer_capacity'] = max(buffer_capacity_error, errors['buffer_capacity'])
                    machine_capacity_error = calculate_error(graph.nodes[station]['machine_capacity'],
                                                             truth.nodes[station]['machine_capacity'])
                    errors['machine_capacity'] = max(machine_capacity_error, errors['machine_capacity'])
                    processing_time_mean_error = calculate_error(graph.nodes[station]['processing_time']['mean'],
                                                                 truth.nodes[station]['processing_time']['mean'])
                    errors['processing_time']['mean'] = max(processing_time_mean_error, errors['processing_time']['mean'])
                    processing_time_std_error = calculate_error(graph.nodes[station]['processing_time']['std'],
                                                                truth.nodes[station]['processing_time']['std'])
                    errors['processing_time']['std'] = max(processing_time_std_error, errors['processing_time']['std'])

                errors['routing_probability'] = 0.0
                errors['transfer_time'] = dict()
                errors['transfer_time']['mean'] = 0.0
                errors['transfer_time']['std'] = 0.0
                connections = graph.edges.keys()
                for connection in connections:
                    routing_probability_error = calculate_error(graph.edges[connection]['routing_probability'],
                                                                truth.edges[connection]['routing_probability'])
                    errors['routing_probability'] = max(routing_probability_error, errors['routing_probability'])
                    transfer_time_mean_error = calculate_error(graph.edges[connection]['transfer_time']['mean'],
                                                               truth.edges[connection]['transfer_time']['mean'])
                    errors['transfer_time']['mean'] = max(transfer_time_mean_error, errors['transfer_time']['mean'])
                    transfer_time_std_error = calculate_error(graph.edges[connection]['transfer_time']['std'],
                                                              truth.edges[connection]['transfer_time']['std'])
                    errors['transfer_time']['std'] = max(transfer_time_std_error, errors['transfer_time']['std'])

                errors['wip_limit'] = calculate_error(graph.graph['wip_limits'][0], limit)

                average_errors[instance][limit][length]['buffer_capacity'] += errors['buffer_capacity']
                average_errors[instance][limit][length]['machine_capacity'] += errors['machine_capacity']
                average_errors[instance][limit][length]['processing_time']['mean'] += errors['processing_time']['mean']
                average_errors[instance][limit][length]['processing_time']['std'] += errors['processing_time']['std']
                average_errors[instance][limit][length]['routing_probability'] += errors['routing_probability']
                average_errors[instance][limit][length]['transfer_time']['mean'] += errors['transfer_time']['mean']
                average_errors[instance][limit][length]['transfer_time']['std'] += errors['transfer_time']['std']
                average_errors[instance][limit][length]['wip_limit'] += errors['wip_limit']

            average_errors[instance][limit][length]['buffer_capacity'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['machine_capacity'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['processing_time']['mean'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['processing_time']['std'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['routing_probability'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['transfer_time']['mean'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['transfer_time']['std'] /= NUMBER_OF_REPLICAS
            average_errors[instance][limit][length]['wip_limit'] /= NUMBER_OF_REPLICAS


# Draw the average maximum errors of attribute estimates
has_legend = False
for instance in SYSTEM_INSTANCES:
    for limit in WIP_LIMITS:
        plt.xlim([LOG_LENGTHS[0], LOG_LENGTHS[-1]])
        plt.xscale('symlog')
        plt.xlabel('Log Length')
        plt.ylim([0.0, 1.0])
        plt.yticks([0.1 * q for q in range(11)])
        plt.ylabel('Average Maximum Error')
        plt.grid(True, linestyle=(0, (1, 10)))
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['buffer_capacity'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['machine_capacity'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['processing_time']['mean'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['processing_time']['std'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['routing_probability'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['transfer_time']['mean'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['transfer_time']['std'] for length in LOG_LENGTHS])
        plt.plot(LOG_LENGTHS, [average_errors[instance][limit][length]['wip_limit'] for length in LOG_LENGTHS])
        if not has_legend:
            plt.legend([r'$b_{s}$', r'$m_{s}$', r'$\mu(P_{s})$', r'$\sigma(P_{s})$', r'$\beta_{s}$', r'$\mu(T_{s,s\prime})$',
                        r'$\sigma(T_{s,s\prime})$', r'$n_{\mathrm{max}}$'], title='Attribute', ncols=2)
            has_legend = True
        figure_name = instance[:instance.find('-')] + '_wip' + str(limit) + '_errors.eps'
        figure_path = os.path.join(FIGURES_FOLDER, figure_name)
        plt.savefig(figure_path, format='eps', bbox_inches='tight')
        plt.show(block=True)
