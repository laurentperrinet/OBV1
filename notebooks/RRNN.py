#! /usr/bin/env python
# -*- coding: utf-8 -*-

#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

import pandas as ps
import numpy as np
import matplotlib.pyplot as plt

import time

import pyNN.nest as sim
#import pyNN.brian as sim

from pyNN.parameters import Sequence
from pyNN.random import RandomDistribution as rnd
from pyNN.random import NumpyRNG as rng

from pyNN.common.populations import PopulationView

from pyNN.utility.plotting import Figure, Panel

from NeuroTools.signals.spikes import SpikeTrain
import os

from IPython.display import display
from ipywidgets import interact

class RRNN:
    def __init__(self, ring=True, recurrent=False, seed=56, source='poisson'):

        #==================================================
        #==============  Parameters =======================
        #==================================================

        #The RRNN is a ring recurrent by default


#         self.simtime = time
        self.seed = seed
        self.source = source
        self.ring, self.recurrent = ring, recurrent
        self.default_params()


    def default_params(self, time=100, N=1080, n_model='cond_exp', i_rate=10., s=1., p=.5):
        self.N = N                      #number of neurons in the network
        #self.N_show = 40                           #Number of neurons prompted in rasterplots



        #c : connectivity sparseness ("all to all by default in recurrent ring")
        #w : global weight, value for every connections
        #i_rate : input rate, mean firing rate of source population neurons
        #w_in : weight of connections between source population and excitatory neurons population
        #s : synaptic delay
        #g : inhibition-excitation coupling
        #p : excitatory neurons percentage in network
        #n_model : neuron dynamic model
        #b_input : input orientation distribution bandwidth
        #angle_input : the most represented orientation angle in input distribution
        #b_xx : orientation selectivity for a projection xx

        #RRNN SR state: 
        #w = 3.4,
        #wie = .5,
        #g = 2.,
        #c = 0.15,
        #p = .5,
        #i_rate = 10,
        #s=1.
        if self.ring:
            self.c = 1
            if self.recurrent :
                self.w, w_input_exc = .5, .5
                self.g = 2.
            else:
                self.w, w_input_exc = .1, .5
                self.g = 0.
        else:
            self.c = 0.15
            if self.recurrent :
                self.w, w_input_exc = .9, .5
                self.g = 2.
            else:
                self.w, w_input_exc = 0, .5
                self.g = 0.
        #------- Cell's parameters -------
        self.cell_params = {
        'tau_m'      : 20.0,   # (ms)
        'tau_syn_E'  : 2.0,    # (ms)
        'tau_syn_I'  : 4.0,    # (ms)
        'e_rev_E'    : 0.0,    # (mV)
        'e_rev_I'    : -70.0,  # (mV)
        'tau_refrac' : 2.0,    # (ms)
        'v_rest'     : -60.0,  # (mV)
        'v_reset'    : -70.0,  # (mV)
        'v_thresh'   : -50.0,  # (mV)
        'cm'         : 0.5,    # (nF)
        }

        self.sim_params = {

        'simtime'     : time,       # (ms)
        #'dt'          : 0.1,               # (ms)

        'input_rate'  : i_rate,                # (Hz)
        'b_input'     : np.inf,
        'angle_input' : 90, # degrees


        'nb_neurons'  : self.N,    #neurons number
        'p'           : p,        #excitators rate in the population
        'neuron_model': n_model,    #the neuron model
        'v_init_min'   : -53.5,  # (mV)
        'v_init_max'   : -49.75,  # (mV)
        #connectivity
        #'c_input_exc' : 1., #self.c*10,
        'c_input_inh' : 0, #self.c*0.,
        'w_input_exc' : w_input_exc,
        #synaptic delay (ms)
        's_input_exc' : s,
        's_input_inh' : s,
        's_exc_inh'   : s,
        's_inh_exc'   : s,
        's_exc_exc'   : s,
        's_inh_inh'   : s,
        #B_thetas for the ring
        # 'b_input_exc' : np.inf,
        'b_exc_inh'   : np.inf,
        'b_exc_exc'   : np.inf,
        'b_inh_exc'   : np.inf,
        'b_inh_inh'   : np.inf
        }
        self.init_params()

        if self.ring :
            self.sim_params['b_input'] = 10.
            self.sim_params['b_exc_inh'] = 50.
            self.sim_params['b_exc_exc'] = 4.
            self.sim_params['b_inh_exc'] = 50.
            self.sim_params['b_inh_inh'] = 4.

    def init_params(self):
        self.sim_params.update({
        'c_exc_inh'   : self.c,
        'c_inh_exc'   : self.c,
        'c_exc_exc'   : self.c,
        'c_inh_inh'   : self.c,

        #synaptic weight (µS)
        'w_exc_inh'   : self.w*self.g,
        'w_inh_exc'   : self.w,
        'w_exc_exc'   : self.w,
        'w_inh_inh'   : self.w*self.g,
        })
    #============================================
    #=========== The Network ====================
    #============================================
    def model(self, sim_index=0, sim_params=None, cell_params=None):
        if sim_params is None :
            sim_params = self.sim_params

        if cell_params is None:
            cell_params = self.cell_params

        # === Build the network =========================================================

        sim.setup(timestep=0.1)#, threads=4)#dt=sim_params['dt'])

        python_rng = rng(seed=self.seed)

        #--setting up nodes and neurons--

        #tuning_function = lambda i, j, B, N : np.exp((np.cos(2.*((i-j)/N*np.pi))-1)/(B*np.pi/180)**2)
        N_in = int(sim_params['nb_neurons']*sim_params['p'])
        self.spike_source = sim.Population(N_in, sim.SpikeSourcePoisson(rate=sim_params['input_rate']))

        if True: #not sim_params['b_input'] == np.inf:
            angle = 1. * np.arange(N_in)
            rates = self.tuning_function(angle, sim_params['angle_input']/180.*N_in, sim_params['b_input'], N_in)
            rates /= rates.mean()
            rates *= sim_params['input_rate']
            # print(rates)
            for i, cell in enumerate(self.spike_source):
                cell.set_parameters(rate=rates[i])


        if sim_params['neuron_model'] == 'cond_exp':
            model = sim.IF_cond_exp
        elif sim_params['neuron_model'] == 'cond_alpha':
            model = sim.IF_cond_alpha

        E_neurons = sim.Population(N_in,
                                   model(**cell_params),
                                   initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
                                   label="NE")

        I_neurons = sim.Population(sim_params['nb_neurons'] - N_in,
                                   model(**cell_params),
                                   initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
                                   label="NI")
        #
        # if sim_params['neuron_model'] == 'cond_alpha':
        #     E_neurons = sim.Population(int(sim_params['nb_neurons'] * sim_params['p']),
        #                                sim.IF_cond_alpha(**cell_params),
        #                                initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
        #                                label="NE")
        #
        #     I_neurons = sim.Population(sim_params['nb_neurons'] - int(sim_params['nb_neurons'] * sim_params['p']),
        #                                sim.IF_cond_alpha(**cell_params),
        #                                initial_values={'v': rnd('uniform', (sim_params['v_init_min'], sim_params['v_init_max']))},
        #                                label="NI")

         #--Setting up connections and optional injections--
        if self.source == 'sweep':
            sweep = sim.DCSource(amplitude=0.1, start=250.0, stop=500.0)
            sweep.inject_into(E_neurons)

        input_exc = sim.Projection(self.spike_source, E_neurons,
                                #connector=sim.FixedProbabilityConnector(sim_params['c_input_exc'], rng=python_rng),
                                #synapse_type=syn['input_exc'],
                                #receptor_type='excitatory')
                                sim.OneToOneConnector(),
                                sim.StaticSynapse(weight=sim_params['w_input_exc'], delay=sim_params['s_input_exc'])
                                )
                                # syn['input_exc'])

        conn_types = ['exc_inh', 'inh_exc', 'exc_exc', 'inh_inh']   #connection types

        syn = {}
        proj = {}

        for conn_type in conn_types :

            weight = sim_params['w_{}'.format(conn_type)]
            delay=sim_params['s_{}'.format(conn_type)]
            syn[conn_type] = sim.StaticSynapse(delay=delay)#weight=weight,
            if conn_type[:3]=='exc':
                pre_neurons = E_neurons
                receptor_type='excitatory'
            else:
                pre_neurons = I_neurons
                receptor_type='inhibitory'
            if conn_type[-3:]=='exc':
                post_neurons = E_neurons
            else:
                post_neurons = I_neurons

            sparseness = sim_params['c_{}'.format(conn_type)]
            proj[conn_type]  = sim.Projection(pre_neurons, post_neurons,
                                            connector=sim.FixedProbabilityConnector(sparseness, rng=python_rng),
                                            synapse_type=syn[conn_type],
                                            receptor_type=receptor_type)

            bw = sim_params['b_{}'.format(conn_type)]
            angle_pre = 1. * np.arange(proj[conn_type].pre.size)
            angle_post = 1. * np.arange(proj[conn_type].post.size)
            w_ij = self.tuning_function(angle_pre[:, np.newaxis], angle_post[np.newaxis, :], bw, N_in)*weight
            proj[conn_type].set(weight=w_ij)

        # exc_inh = sim.Projection(E_neurons, I_neurons,
        #                         connector=sim.FixedProbabilityConnector(sim_params['c_exc_inh'], rng=python_rng),
        #                         synapse_type=syn['exc_inh'],
        #                         receptor_type='excitatory')
        #
        # inh_exc = sim.Projection(I_neurons, E_neurons,
        #                         connector=sim.FixedProbabilityConnector(sim_params['c_inh_exc'], rng=python_rng),
        #                         synapse_type=syn['inh_exc'],
        #                         receptor_type='inhibitory')
        #
        # exc_exc = sim.Projection(E_neurons, E_neurons,
        #                         connector=sim.FixedProbabilityConnector(sim_params['c_exc_exc'], rng=python_rng),
        #                         synapse_type=syn['exc_exc'],
        #                         receptor_type='excitatory')
        #
        # inh_inh = sim.Projection(I_neurons, I_neurons,
        #                         connector=sim.FixedProbabilityConnector(sim_params['c_inh_inh'], rng=python_rng),
        #                         synapse_type=syn['inh_inh'],
        #                         receptor_type='inhibitory')
        #
        # v = locals()
        # for conn_type in conn_types :
        #     proj = v['{}'.format(conn_type)]
        #     if not bw == np.inf:
        #         angle_pre = 1. * np.arange(proj.pre.size)
        #         angle_post = 1. * np.arange(proj.post.size)
        #         w = tuning_function(angle_pre[:, np.newaxis], angle_post[np.newaxis, :], bw, N_in)*w
        #         proj.set(weight=w)
        #

        #--setting up recording--
        self.spike_source.record('spikes')
        E_neurons.record('spikes')
        I_neurons.record('spikes')

        # === Run simulation ============================================================
        sim.run(sim_params['simtime'])

        # === Save ROI data and CV computing ============================================
        spikesE = E_neurons.get_data().segments[0]
        spikesI = I_neurons.get_data().segments[0]
        self.spikesP = self.spike_source.get_data().segments[0]

        self.spikesE = spikesE
        self.spikesI = spikesI

        #------- computing cv -------
        all_CVs = np.array([])
        for st in spikesE.spiketrains :
            all_CVs = np.append(all_CVs, SpikeTrain(np.array(st)).cv_isi())
        for st in spikesI.spiketrains :
            all_CVs = np.append(all_CVs, SpikeTrain(np.array(st)).cv_isi())
        #-----------------------------

        megadico = sim_params.copy()
        megadico.update(cell_params.copy())
        megadico.update({'m_f_rateE': E_neurons.mean_spike_count()})
        megadico.update({'m_f_rateI': I_neurons.mean_spike_count()})
        megadico.update({'m_f_rate' : (E_neurons.mean_spike_count()*sim_params['p'] + I_neurons.mean_spike_count()*(1-sim_params['p']))*1000.0/sim_params['simtime']})
        megadico.update({'cv' : np.nanmean(all_CVs)})

        # === Clearing and return data ==================================================
        sim.end()
        df = ps.DataFrame(data = megadico, index = [sim_index])
        return df, spikesE, spikesI

    #========================================
    #============= Ring tuning ==============
    #========================================

    def tuning_function(self, i, j, B, N):
        if B==np.inf:
            VM = np.ones_like(i*j)
        else:
            VM = np.exp((np.cos(2.*((i-j)/N*np.pi))-1)/(B*np.pi/180)**2)
        VM /= VM.sum(axis=0)
        return VM


    #========================================
    #========= Setting parameters ===========
    #========================================

    def setParams(self, keys, values):
        for i, key in enumerate(keys):
            value = values[i]
            self.sim_params[key] = value

        #print self.sim_params


    #========================================
    #======= Rasterplotting functions =======
    #========================================
    #----- One parameter variation ------
    def Raster(self, df_sim, spikesE, spikesI, title=' RRNN ', input=True, markersize=.5):
        #eventplot
        # line_properties = {'c':'r'}
        if input:
            f = Figure(
                    Panel(self.spikesP.spiketrains, xticks=False, ylabel="input", color='k', markersize=markersize), #, line_properties=line_properties
                    Panel(spikesE.spiketrains, xticks=False, ylabel="Excitatory", color='r', markersize=markersize),
                    Panel(spikesI.spiketrains, xlabel="Time (ms)", xticks=True, color='b', ylabel="Inhibitory", markersize=markersize),
                    title='--------- {} ---------'.format(title))

        else:
            f = Figure(
                    Panel(spikesE.spiketrains[0:self.N_show], xticks=False, ylabel="Excitatory"),
                    Panel(spikesI.spiketrains[0:self.N_show], xlabel="Time (ms)", xticks=True, ylabel="Inhibitory"),
                    title='--------- {} ---------'.format(title))

        for ax in f.fig.axes:
            ax.set_axis_bgcolor('w')
            ax.set_xticks(np.linspace(0, self.sim_params['simtime'], 6, endpoint=True))
        f.fig.subplots_adjust(hspace=0) # TODO
        return f

    #----- One parameter variation ------
    def variationRaster(self, var_name, values, force_int=False):
        df = None
        if force_int :
            values = [int(i) for i in values]

        sim_params = self.sim_params.copy()
        for i, value in enumerate(values):
            sim_params[var_name] = value
            df_sim, spikesE, spikesI = self.model(i, sim_params)

            if False: #var_name[0] == 'w':
                _ = self.Raster(df_sim, spikesE, spikesI,
                            title=' {0} = {1} '.format(var_name, str(value/self.w) + ' w'))
            else :
                _ = self.Raster(df_sim, spikesE, spikesI,
                            title=' {0} = {1} '.format(var_name, value))

            plt.show()

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)

        return df

    #----- Two parameters, same variation on each -------
    def variationRaster_twoParams(self, var1_name, var2_name, values, force_int=False):
        df = None
        if force_int :
            values = [int(i) for i in values]

        sim_params = self.sim_params.copy()
        for i, value in enumerate(values):
            sim_params[var1_name] = value
            sim_params[var2_name] = value

            df_sim, spikesE, spikesI = self.model(i, sim_params)

            _ = self.Raster(df_sim, spikesE, spikesI,
                        title='--------- {0} = {1} = {2} ---------'.format(var1_name, var2_name, value))

            plt.show()

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)

        return df

    #----- Two parameters, different variation on each -------
    def doubleVariationRaster(self, var1_name, values1, var2_name, values2, force_int=False):
        idx = 0
        df = None

        if force_int :
            values1 = [int(i) for i in values1]

        sim_params = self.sim_params.copy()
        for value1 in values1:
            sim_params[var1_name] = value1

            for value2 in values2:
                sim_params[var2_name] = value2
                df_sim, spikesE, spikesI = self.model(idx, sim_params)

                _ = self.Raster(df_sim, spikesE, spikesI,
                        title='--------- {0} = {1} & {2} = {3} ---------'.format(var1_name, value1, var2_name, value2))

                plt.show()

                if df is None:
                    df = df_sim
                else:
                    df = df.append(df_sim)
                idx += 1
        return df

    def doubleVariationRaster_P2P(self, var1_name, values1, var2_name, values2, force_int=False):
        df = None
        if force_int :
            values1 = [int(i) for i in values1]

        sim_params = self.sim_params.copy()
        for i, value1 in enumerate(values1):
            value2 = values2[i]
            sim_params[var1_name] = value1
            sim_params[var2_name] = value2

            df_sim, spikesE, spikesI = self.model(i, sim_params)

            _ = self.Raster(df_sim, spikesE, spikesI,
                        title='--------- {0} = {1} & {2} = {3} ---------'.format(var1_name, value1, var2_name, value2))

            plt.show()

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)

        return df

#====================================================
#=============   I/O rate functions =================
#====================================================

    def variationDF(self, var_name, values, force_int=False):
        df = None
        if force_int :
            values = [int(i) for i in values]

        sim_params = self.sim_params.copy()
        cell_params = self.cell_params.copy()

        for i, value in enumerate(values):
            if var_name in sim_params.keys():
                sim_params[var_name] = value
            else :
                cell_params[var_name] = value

            df_sim, stash, stash1 = self.model(i, sim_params, cell_params)

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)
        #print(df[var_name])
        return df

    def variation_twoParamsDF(self, var1_name, var2_name, values, df=None):
        sim_params = self.sim_params.copy()

        for i, value in enumerate(values) :
            sim_params[var1_name] = value
            sim_params[var2_name] = value

            df_sim, stash, stash1 = self.model(i, sim_params)

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)

        return df

    def doubleVariationDF(self, var1_name, var2_name, values1, values2, df=None):
        sim_params = self.sim_params.copy()

        for i, value1 in enumerate(values1) :
            sim_params[var1_name] = value1
            sim_params[var2_name] = values2[i]

            df_sim, stash, stash1 = self.model(i, sim_params)

            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)

        return df


    def paramRole(self, sim_list, f_rate_max=400, datapath='data_RRNN', semilog = True):
        tag= 'tmp' + str(int(time.time()))
        for param_name, param_range in sim_list:

            try:
                os.makedirs(datapath)
            except: pass
            #print(os.path.join(os.getcwd(), datapath))
            filename = datapath + '/'+ param_name + tag + '.pkl'

            print ('------------  {0}  -------------'.format(param_name))

            try :
                df = ps.read_pickle(filename)
            except :
                df = self.variationDF(param_name, param_range)
                df.to_pickle(filename)


            plt.figure(figsize=(15,5))
            if semilog :
                plt.semilogx(df[param_name], df['m_f_rate'], '-', lw=2)
                plt.semilogx(df[param_name], df['m_f_rate'], 'r.', lw=2)
            else :
                plt.plot(df[param_name], df['m_f_rate'], '-.')
            plt.xlabel(param_name)
            plt.ylabel("Output rate (Hz)")
            plt.axis('tight')
            if f_rate_max is None:
                plt.ylim(0)
            else:
                plt.ylim([0, f_rate_max])
            plt.show()


    def covariation_Curve(self, var1_name, var2_name, values, datapath, semilog = False, g = True, f_rate_max = 400):
        try :
            os.makedirs(datapath)
        except : pass

        filename = datapath + '/' + var1_name + var2_name + '.pkl'

        if g:
            print ('------------ G -------------')
        else :
            print  ('----------- {0}, {1} ------------'.format(var1_name, var2_name))

        try :
            df = ps.read_pickle(filename)
        except :
            df = self.variation_twoParamsDF(var1_name, var2_name, values)
            df.to_pickle(filename)

        plt.figure(figsize=(15,5))

        if g :
            if semilog :
                plt.semilogx(df[var1_name]/self.w, df['m_f_rate'], '-.', lw=2)
            else :
                plt.plot(df[var1_name]/self.w, df['m_f_rate'], '-.')

            plt.xlabel('g')
        plt.ylabel("Output rate (Hz)")
        plt.ylim([0, f_rate_max])
        plt.show()

#======================================================
#============ Dynamic caracterisations ================
#======================================================

    def plot_isi_hist(self, panel, segment, label, hide_axis_labels=False):
        print("plotting ISI histogram (%s)" % label)
        bin_width = 0.2
        bins_log = np.arange(0, 8, 0.2)
        bins = np.exp(bins_log)
        all_isis = np.concatenate([np.diff(np.array(st)) for st in segment.spiketrains])
        isihist, bins = np.histogram(all_isis, bins)
        xlabel = "Inter-spike interval (ms)"
        ylabel = "n in bin"
        if hide_axis_labels:
            xlabel = None
            ylabel = None
        plot_hist(panel, isihist, bins_log, bin_width, label=label,
            xlabel=xlabel, xticks=np.log([10, 100, 1000]),
            xticklabels=['10', '100', '1000'], xmin=np.log(2),
            ylabel=ylabel)

    #-----Maybe the algorithm is too heavy--------
    def spikeCount(self, segments):
        simtime = self.sim_params['simtime']

        sumSpikes = np.zeros(simtime*10+1)
        for segment in segments :
            for spiketrain in segment.spiketrains :
                st = np.array(spiketrain)
                for t in st :
                    sumSpikes[t*10] += 1


    def plot_spikeCount(self, segments):
        simtime = self.sim_params['simtime']

        sumSpikes = np.zeros(simtime*10+1)
        for segment in segments :
            for spiketrain in segment.spiketrains :
                st = np.array(spiketrain)
                for t in st :
                    sumSpikes[t*10] += 1

        timeline = np.linspace(0, simtime, simtime*10+1)

        plt.figure(1)
        plt.plot(timeline, sumSpikes, '-')
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike number')
        plt.show()
#
#     def g_dFoverdI(self, g_values):
#         df = None
#
#         for value in g_values:
#             self.setParams(['w_inh_exc', 'w_inh_inh'], [value, value])
#             #self.setParams(['w_exc_inh', 'w_inh_inh'], [value, value])
#             df_sim = self.variationDF('input_rate', self.sim_params['input_rate']*np.logspace(-.02, .02, 3, endpoint=True))
#             if df is None:
#                 df = df_sim
#             else:
#                 df = df.append(df_sim)
#         return df
#
#     def c_dFoverdI(self, c_values):
#         df = None
#
#         for value in c_values:
#             self.setParams(['c_exc_inh', 'c_exc_exc', 'c_inh_exc', 'c_inh_inh'],
#                             [value, value, value, value])
#             df_sim = self.variationDF('input_rate', self.sim_params['input_rate']*np.logspace(-.02, .02, 3, endpoint=True))
#             if df is None:
#                 df = df_sim
#             else:
#                 df = df.append(df_sim)
#         return df
#
#     def w_dFoverdI(self, w_values):
#         df = None
#
#         for value in w_values:
#             self.setParams(['w_exc_exc', 'w_exc_inh', 'w_inh_exc', 'w_inh_inh'],
#                             [value, value, self.g*value, self.g*value])
#             df_sim = self.variationDF('input_rate', self.sim_params['input_rate']*np.logspace(-.02, .02, 3, endpoint=True))
#             if df is None:
#                 df = df_sim
#             else:
#                 df = df.append(df_sim)
#         return df
#
#     def win_dFoverdI(self, win_values):
#         df = None
#
#         for value in win_values:
#             self.setParams(['w_input_exc'],
#                             [value])
#             df_sim = self.variationDF('input_rate', self.sim_params['input_rate']*np.logspace(-.02, .02, 3, endpoint=True))
#             if df is None:
#                 df = df_sim
#             else:
#                 df = df.append(df_sim)
#         return df


    def dFoverdI(self, values, var):
        df = None

        for value in values:
            if var == 'g':
                self.g = value
            elif var == 'c':
                self.c = value
            elif var == 'w':
                self.w = value
            self.init_params()
            df_sim = self.variationDF('input_rate', self.sim_params['input_rate']*np.logspace(-.02, .02, 3, endpoint=True))
            if df is None:
                df = df_sim
            else:
                df = df.append(df_sim)
        return df

    def multiOptimisation(self, values, var='g', datapath = os.path.join(os.getcwd(), 'data_BalancedRRNN')):

        try:
            os.mkdir(datapath)
        except:
            pass

#         #-------  G  --------
#         if c_or_w is 'g':
#             filename = os.path.join(os.getcwd(), datapath, 'DataG.pkl')
#             try:
#                 df = ps.read_pickle(filename)
#             except:
#                 df = self.g_dFoverdI(values)
#                 df.to_pickle(filename)
#             print(self.value_minCost(df, len(values), 'g'))
#         #-----  Sparseness ------
#         elif c_or_w == 'c' :
#             filename = os.path.join(os.getcwd(), datapath, 'DataSpars.pkl')
#             try:
#                 df = ps.read_pickle(filename)
#             except:
#                 df = self.c_dFoverdI(values)
#                 df.to_pickle(filename)
#             print (self.value_minCost(df, len(values), 'c_exc_inh'))
#         #------  Weight  -------
#         elif c_or_w == 'w' :
#             filename = os.path.join(os.getcwd(), datapath, 'DataWeight.pkl')
#             try:
#                 df = ps.read_pickle(filename)
#             except:
#                 df = self.w_dFoverdI(values)
#                 df.to_pickle(filename)
#             print( self.value_minCost(df, len(values), 'w_exc_inh'))
        filename = os.path.join(datapath, 'DataWeight.pkl')
        try:
            df = ps.read_pickle(filename)
        except:
            df = self.dFoverdI(values, var=var)
            df.to_pickle(filename)
        print('Optimum at ', self.value_minCost(df, values, var))
        return df
        #------- Input weight -----
#         elif c_or_w == 'win' :
#             filename = os.path.join(os.getcwd(), datapath, 'DataWeightIn.pkl')
#             try:
#                 df = ps.read_pickle(filename)
#             except:
#                 df = self.win_dFoverdI(values)
#                 df.to_pickle(filename)
#             print (self.value_minCost(df, len(values), 'w_input_exc'))
#         else:
#             print('fâdâ va ')

    def value_minCost(self, df, values, var, dfdI_norm=10, lambda_cv=.8, sigma_cv=.5):
        n = len(values)
        dI0, dI1, dI2 = np.array(df['input_rate'])[0], np.array(df['input_rate'])[1], np.array(df['input_rate'])[2]
        fr = np.array(df['m_f_rate'])
        cv = np.array(df['cv'])
        cv = cv.reshape((n, 3))
#         param_value = np.array(df[var])
#         pv = param_value.reshape((n, 3))
#         N_pv = np.size(pv)
        fr = fr.reshape((n, 3))
        dfdI = ((fr[:, 1] - fr[:, 0]) / (dI1-dI0) + (fr[:, 2] - fr[:, 1]) / (dI2-dI1)) * .5
        cost = (1-lambda_cv) * (1 - dfdI / dfdI.max()) + lambda_cv * (1- np.exp(-.5*(1-cv[:, 1])**2/sigma_cv**2))
#         if var == 'g':
#             fig, ax = plt.subplots(figsize=(13, 5))
#             ax.plot(pv[:,1]/self.w, cv[:,1], label='CV')
#             ax.plot(pv[:,1]/self.w, dfdI / dfdI_norm, label='sensitivity')
#             ax.legend()
#
#             fig, ax = plt.subplots(figsize=(13, 5))
#             ax.plot(pv[:,1]/self.w, 1 - np.exp(-.5*(1-cv[:,1])**2/sigma_cv**2), label='poissonness')
#             ax.plot(pv[:,1]/self.w, 1 - dfdI / dfdI.max(), label='inv. sensit.')
#             ax.plot(pv[:,1]/self.w, cost, label='total cost')
#             ax.legend()
#             plt.tight_layout()
#
#             ind = np.argmin(cost)
#             return pv[ind][0]/self.w
#
#         else:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(values, cv[:, 1], label='CV')
        ax.plot(values, dfdI / 100., label='sensitivity')
        ax.legend()

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(values, 1 - np.exp(-.5*(1-cv[:, 1])**2/sigma_cv**2), label='poissonness')
        ax.plot(values, 1 - dfdI / dfdI.max(), label='inv. sensit.')
        ax.plot(values, cost, label='total cost')
        ax.legend()
        plt.tight_layout()

        ind = np.argmin(cost)
        return values[ind]

    #======================================================
    #================  Miscellaneous ======================
    #======================================================


    #---------Fitting network activity with Von mises distribution----------
    def fit_vonMises(self, spikes, verbose=False):
            theta = self.sim_params['angle_input']*np.pi/180
            fr = np.zeros(len(spikes.spiketrains))
            for i, st in enumerate(spikes.spiketrains):
                fr[i] = np.float(len (st))

            def mises(x, sigma, amp, m=np.pi/2):
                kappa = 1. / sigma**2
                exp_c = np.exp(np.cos(2*(x-m))*kappa)
                return amp * exp_c #/(2*np.pi*iv(0, kappa))

            from lmfit import Model
            vonM_mod = Model(mises)
            #vonM_mod.param_names
            #vonM_mod.independent_vars

            y = np.array(fr)
            x = np.linspace(0, np.pi, len(spikes.spiketrains))
            result = vonM_mod.fit(y, x = x, sigma=np.pi/2, amp=y.mean(), m=np.pi/2)
            if verbose: print(result.fit_report())
            return x, y, result

    #----------Plotting histograms-------------
    def plot_hist(self, panel, hist, bins, width, xlabel=None, ylabel=None,
                    label=None, xticks=None, xticklabels=None, xmin=None, ymax=None):
            if xlabel: panel.set_xlabel(xlabel)
            if ylabel: panel.set_ylabel(ylabel)
            for t,n in zip(bins[:-1], hist):
                panel.bar(t, n, width=width, color=None)
            if xmin: panel.set_xlim(xmin=xmin)
            if ymax: panel.set_ylim(ymax=ymax)
            if xticks is not None: panel.set_xticks(xticks)
            if xticklabels: panel.set_xticklabels(xticklabels)
            panel.text(0.8, 0.8, label, transform=panel.transAxes)
