import matplotlib.pyplot as pyplt
import math

import characteristics_utils as ch_utils
import plot_utils as plt_utils
from distributions import dist_list


def lab_1_part_hist():
    sample_size_seq = (
        10,
        100,
        1000
    )
    experiment_list = (
        {
            'dist': dist_list['norm'],
            'min_range': (-5, 5),
            'func_points_count': 100,
            'bins_mode': 'width',
            'bins_width': 0.5
        },
        {
            'dist': dist_list['cauchy'],
            'min_range': (-25, 25),
            'func_points_count': 1000,
            'bins_mode': 'count',
            'bins_count': 16
        },
        {
            'dist': dist_list['laplace'],
            'min_range': (-5, 5),
            'func_points_count': 100,
            'bins_mode': 'count',
            'bins_count': 16
        },
        {
            'dist': dist_list['poisson'],
            'min_range': (0, 12),
            'func_points_count': -1,
            'bins_mode': 'count',
            'bins_count': 10},
        {
            'dist': dist_list['uniform'],
            'min_range': (-3, 3),
            'func_points_count': 100,
            'bins_mode': 'width',
            'bins_width': 0.4
        }
    )

    for experiment in experiment_list:
        fig, axis = pyplt.subplots(1, len(sample_size_seq), figsize=(11, 3),
                                   gridspec_kw={'left': 0.06, 'right': 0.990, 'bottom': 0.16, 'wspace': 0.3})
        for i in range(len(sample_size_seq)):
            sample_size = sample_size_seq[i]
            sample = experiment['dist']['get_sample'](sample_size)
            plot_range = plt_utils.upscale_range_to_sample(sample, experiment['min_range'])
            if experiment.get('bins_mode') == 'count':
                plt_utils.plot_sample_hist(axis[i], plot_range, sample, bins_count=experiment['bins_count'])
            elif experiment.get('bins_mode') == 'width':
                plt_utils.plot_sample_hist(axis[i], plot_range, sample, bins_width=experiment['bins_width'])
            else:
                plt_utils.plot_sample_hist(axis[i], plot_range, sample)
            plt_utils.plot_func(axis[i], plot_range, experiment['func_points_count'],
                                experiment['dist']['density_func'])
            axis[i].set_title('sample size = ' + str(sample_size))
            axis[i].set_xlabel('Value')
            axis[i].set_ylabel('Density')
        fig.savefig('lab_1/part_hist/figures/' + experiment['dist']['name'] + '.png')


def lab_1_part_table():
    main_sample_size = 1000
    sample_size_seq = (
        10,
        100,
        1000
    )
    experiment_list = (
        {
            'dist': dist_list['norm'],
        },
        {
            'dist': dist_list['cauchy'],
        },
        {
            'dist': dist_list['laplace'],
        },
        {
            'dist': dist_list['poisson'],
        },
        {
            'dist': dist_list['uniform'],
        }
    )
    stat_characteristics_list = (
        ch_utils.data_mean,
        ch_utils.seq_median,
        ch_utils.sample_z_r,
        ch_utils.seq_z_q,
        ch_utils.seq_z_tr
    )
    file = open('lab_1/part_table/characteristics_table.txt', 'w')
    for experiment in experiment_list:
        file.write('\\hline\n')
        file.write('{0} & & & & & \\\\\n'.format(experiment['dist']['name']))
        file.write('\\hline\n')
        for sample_size in sample_size_seq:
            characteristics_table = [[] for characteristic in stat_characteristics_list]
            for i in range(main_sample_size):
                seq = sorted(experiment['dist']['get_sample'](sample_size))
                for j in range(len(stat_characteristics_list)):
                    char_value = stat_characteristics_list[j](seq)
                    characteristics_table[j].append(char_value)
            char_mean_list = []
            char_dispersion_list = []
            for j in range(len(stat_characteristics_list)):
                mean = ch_utils.data_mean(characteristics_table[j])
                dispersion = ch_utils.data_dispersion(characteristics_table[j])
                char_mean_list.append(round(mean, 6))
                char_dispersion_list.append(round(dispersion, 6))

            file.write('\\hline\n')
            file.write('$size={0}$   &      Mean &    Median &       $z_R$ &      $z_Q$ &      $z_{{tr}}$ \\\\\n'.format(sample_size))
            file.write('\\hline\n')
            file.write('$E(z)$')
            for mean in char_mean_list:
                file.write(' & {0:2.6f}'.format(mean))
            file.write(' \\\\\n')
            file.write('\\hline\n')
            file.write('$D(z)$')
            for dispersion in char_dispersion_list:
                file.write(' & {0:2.6f}'.format(dispersion))
            file.write(' \\\\\n')
            file.write('\\hline\n')
            file.write('$E(z) \\pm \\sqrt{D(z)}$')
            for i in range(len(char_mean_list)):
                file.write(' & [{0:2.6f};'.format(char_mean_list[i] - math.sqrt(char_dispersion_list[i])))
            file.write(' \\\\\n')
            file.write(' ')
            for i in range(len(char_mean_list)):
                file.write(' & {0:2.6f}]'.format(char_mean_list[i] + math.sqrt(char_dispersion_list[i])))
            file.write(' \\\\\n')
            file.write('\\hline\n')
            file.write('$\\widehat{E}(z)$ & ${0.0}^{+0.0}_{-0.0}$ & ${0.0}^{+0.0}_{-0.0}$ & ${0.0}^{+0.0}_{-0.0}$ & '
                       '${0.0}^{+0.0}_{-0.0}$ & ${0.0}^{+0.0}_{-0.0}$\\\\\n')
            file.write('\\hline\n')
        file.write('\n\n')
    file.close()


def lab_1_part_boxplot():
    sample_size_seq = (
        20,
        100
    )
    experiment_list = (
        {
            'dist': dist_list['norm'],
        },
        {
            'dist': dist_list['cauchy'],
        },
        {
            'dist': dist_list['laplace'],
        },
        {
            'dist': dist_list['poisson'],
        },
        {
            'dist': dist_list['uniform'],
        }
    )
    file = open('lab_1/part_boxplot/outliers_table.txt', 'w')
    for experiment in experiment_list:
        fig, axis = pyplt.subplots(1, 1, figsize=(6, 4))
        samples = []
        for i in range(len(sample_size_seq)):
            sample_size = sample_size_seq[i]
            sample = experiment['dist']['get_sample'](sample_size)
            samples.append(sample)
        plt_utils.draw_boxplot(axis, samples, sample_size_seq)
        axis.set_xlabel('Value')
        axis.set_ylabel('Sample size')
        fig.savefig('lab_1/part_boxplot/figures/' + experiment['dist']['name'] + '.png')
        for i in range(len(sample_size_seq)):
            outliers = 0
            sample_size = sample_size_seq[i]
            for j in range(1000):
                sample = experiment['dist']['get_sample'](sample_size)
                outliers += ch_utils.number_of_outliers(sample)
            file.write('{0};{1};{2}\n'.format(experiment['dist']['name'], sample_size, outliers / (sample_size * 1000)))
    file.close()


def lab_1_part_edf():
    sample_size_seq = (
        20,
        60,
        100
    )

    continuous_range = (-4, 4)
    poisson_range = (6, 14)

    experiment_list = (
        {
            'dist': dist_list['norm'],
            'plot_range': continuous_range,
            'func_points_count': 100,
        },
        {
            'dist': dist_list['cauchy'],
            'plot_range': continuous_range,
            'func_points_count': 100,
        },
        {
            'dist': dist_list['laplace'],
            'plot_range': continuous_range,
            'func_points_count': 100,
        },
        {
            'dist': dist_list['poisson'],
            'plot_range': poisson_range,
            'step': True,
            'func_points_count': -1,
        },
        {
            'dist': dist_list['uniform'],
            'plot_range': continuous_range,
            'func_points_count': 100,
        }
    )

    for experiment in experiment_list:
        fig, axis = pyplt.subplots(1, len(sample_size_seq), figsize=(11, 3),
                                   gridspec_kw={'left': 0.06, 'right': 0.990, 'bottom': 0.16, 'wspace': 0.3})
        for i in range(len(sample_size_seq)):
            sample_size = sample_size_seq[i]
            plot_range = experiment['plot_range']
            sample = experiment['dist']['get_sample'](sample_size)
            table_func = ch_utils.get_table_func(sample)
            if experiment.get('step'):
                plt_utils.plot_step_func(axis[i], plot_range, experiment['dist']['func'])
            else:
                plt_utils.plot_func(axis[i], plot_range, experiment['func_points_count'], experiment['dist']['func'])
            plt_utils.plot_table_func(axis[i], plot_range, table_func, 0, 1)
            axis[i].set_title('sample_size = ' + str(sample_size))
            axis[i].set_xlabel('Value')
            axis[i].set_ylabel('Probability')
        fig.savefig('lab_1/part_edf/figures/' + experiment['dist']['name'] + '.png')


def lab_1_part_kde():
    sample_size_seq = (
        20,
        60,
        100
    )
    experiment_list = (
        {
            'dist': dist_list['norm'],
            'plot_range': (-4, 4),
            'func_points_count': 100,
        },
        {
            'dist': dist_list['cauchy'],
            'plot_range': (-20, 20),
            'func_points_count': 100,
        },
        {
            'dist': dist_list['laplace'],
            'plot_range': (-4, 4),
            'func_points_count': 100,
        },
        {
            'dist': dist_list['poisson'],
            'plot_range': (0, 20),
            'func_points_count': -1,
        },
        {
            'dist': dist_list['uniform'],
            'plot_range': (-4, 4),
            'func_points_count': 100,
        }
    )
    factor_list = (0.5, 1, 2)
    for experiment in experiment_list:
        for i in range(len(sample_size_seq)):
            fig, axis = pyplt.subplots(1, len(factor_list), figsize=(11, 3),
                                       gridspec_kw={'left': 0.06, 'right': 0.990, 'bottom': 0.16, 'wspace': 0.3})
            sample_size = sample_size_seq[i]
            plot_range = experiment['plot_range']
            sample = experiment['dist']['get_sample'](sample_size)
            for j in range(len(factor_list)):
                plt_utils.plot_func(axis[j], plot_range, experiment['func_points_count'],
                                    experiment['dist']['density_func'])
                factor = factor_list[j]
                plt_utils.plot_kde(axis[j], sample, factor)
                axis[j].set_xlim(plot_range)
                axis[j].set_title('${{h}_{n}}$ = ' + str(factor))
                axis[j].set_xlabel('Value')
                axis[j].set_ylabel('Density')
            fig.savefig('lab_1/part_kde/figures/' + experiment['dist']['name'] + str(sample_size) + '.png')


def run_lab1():
    lab_1_part_hist()
    pyplt.close('all')
    lab_1_part_edf()
    pyplt.close('all')
    lab_1_part_table()
    pyplt.close('all')
    lab_1_part_boxplot()
    pyplt.close('all')
    lab_1_part_kde()
    pyplt.close('all')