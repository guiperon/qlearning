import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from StochasticGeometry import StochasticGeometry
from SlottedAloha import SlottedAloha_MultipleChannels, SlottedAloha_MultipleChannels_NoNOMA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_MultipleChannels_NoNOMA


def _run_devices_iteration(args):
    (idx, num_dev, Channels_Relays_List, Relays, runs, frames, slots,
     N, r, alpha, gamma, P, Distance, h_Nakagami) = args

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Devices: {num_dev}")

    dist_slice = Distance[:num_dev, :, :]
    h_slice = h_Nakagami[:num_dev, :, :]

    alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(dist_slice)) / 10)
    SNR = (P / N * alpha_k_j * h_slice**2)

    per_channel_results = {}
    for ch_count in Channels_Relays_List:
        tp, dist, total = SlottedAloha_MultipleChannels(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r
        )

        tp_nonoma, dist_nonoma, total_nonoma = SlottedAloha_MultipleChannels_NoNOMA(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r
        )

        QTable = InitializeQTable(num_dev, ch_count, slots, runs, True)
        tp_ql, dist_ql, total_ql = Qlearning_MultipleChannels(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
        )

        QTable = InitializeQTable(num_dev, ch_count, slots, runs, True)
        tp_ql_nonoma, dist_ql_nonoma, total_ql_nonoma = Qlearning_MultipleChannels_NoNOMA(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
        )

        per_channel_results[ch_count] = {
            'SA': (tp, dist, total),
            'SA_NoNOMA': (tp_nonoma, dist_nonoma, total_nonoma),
            'QL': (tp_ql, dist_ql, total_ql),
            'QL_NoNOMA': (tp_ql_nonoma, dist_ql_nonoma, total_ql_nonoma),
        }

    return idx, per_channel_results

# ==============================================================================
# 2. SCRIPT PRINCIPAL (Main)
# ==============================================================================

def run_simulation():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Simulação...")
    
    # %% Parameters
    np.random.seed(0)

    # Devices
    # linspace(start, stop, num)
    Devices = np.linspace(100, 1500, 15, dtype=int)
    
    P_dBm = 10
    P = 10**((P_dBm - 30) / 10)

    FdB = 6
    F = 10**(FdB / 10)
    N0dB = -204
    N0 = 10**(N0dB / 10)
    B = 100e3
    N = N0 * B * F

    # Relays
    Relays = 4
    Channels_Relays_List = [1, 2, 3, 4] # Quantidades de canais a testar

    r = 3
    
    # Stochastic Geometry Parameters
    cell_radius = 5e3

    # Simulation Parameters
    runs = 100 #original 100
    slots = 100
    frames = 50

    alpha = 0.1
    gamma = 0.5

    # Nakagami-m
    m = 2

    # %% Stochastic Geometry
    max_devs = np.max(Devices)
    max_relays = Relays # 4
    
    Distance = StochasticGeometry(max_devs, max_relays, cell_radius, runs)
    
    # Canal Nakagami
    shape_k = m / 2
    scale_theta = 1
    size_h = (max_devs, max_relays, runs)
    
    # MATLAB: sqrt(gamrnd) + j*sqrt(gamrnd)
    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))

    # Estruturas para armazenar resultados
    # Dicionário: results[Metodo][NumCanais] = vetor indexado por cenário de devices
    methods = ['SA', 'SA_NoNOMA', 'QL', 'QL_NoNOMA']
    n_devices = len(Devices)
    res_data = {
        method: {
            ch: {'ntput': np.zeros(n_devices), 'ndist': np.zeros(n_devices), 'ntotal': np.zeros(n_devices)}
            for ch in Channels_Relays_List
        }
        for method in methods
    }

    task_args = [
        (idx, int(num_dev), Channels_Relays_List, Relays, runs, frames, slots,
         N, r, alpha, gamma, P, Distance, h_Nakagami)
        for idx, num_dev in enumerate(Devices)
    ]

    n_workers = min(os.cpu_count() or 1, n_devices)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_devices} device cases.")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_devices_iteration, args): args[0] for args in task_args}
        for future in as_completed(futures):
            idx, per_channel_results = future.result()
            for ch_count in Channels_Relays_List:
                sa_tp, sa_dist, sa_total = per_channel_results[ch_count]['SA']
                sa_no_tp, sa_no_dist, sa_no_total = per_channel_results[ch_count]['SA_NoNOMA']
                ql_tp, ql_dist, ql_total = per_channel_results[ch_count]['QL']
                ql_no_tp, ql_no_dist, ql_no_total = per_channel_results[ch_count]['QL_NoNOMA']

                res_data['SA'][ch_count]['ntput'][idx] = sa_tp
                res_data['SA'][ch_count]['ndist'][idx] = sa_dist
                res_data['SA'][ch_count]['ntotal'][idx] = sa_total

                res_data['SA_NoNOMA'][ch_count]['ntput'][idx] = sa_no_tp
                res_data['SA_NoNOMA'][ch_count]['ndist'][idx] = sa_no_dist
                res_data['SA_NoNOMA'][ch_count]['ntotal'][idx] = sa_no_total

                res_data['QL'][ch_count]['ntput'][idx] = ql_tp
                res_data['QL'][ch_count]['ndist'][idx] = ql_dist
                res_data['QL'][ch_count]['ntotal'][idx] = ql_total

                res_data['QL_NoNOMA'][ch_count]['ntput'][idx] = ql_no_tp
                res_data['QL_NoNOMA'][ch_count]['ndist'][idx] = ql_no_dist
                res_data['QL_NoNOMA'][ch_count]['ntotal'][idx] = ql_no_total

    # %% Processar Dados para Plotagem (Cálculo de Redundância)
    # Redundancy Rate = 100 * (1 - ndist / ntotal)
    
    redundancy = {m: {ch: [] for ch in Channels_Relays_List} for m in methods}
    
    for m in methods:
        for ch in Channels_Relays_List:
            ndist_arr = np.array(res_data[m][ch]['ndist'])
            ntotal_arr = np.array(res_data[m][ch]['ntotal'])
            
            # Evitar div por zero
            with np.errstate(divide='ignore', invalid='ignore'):
                red = 100 * (1 - ndist_arr / ntotal_arr)
                red = np.nan_to_num(red)
            
            redundancy[m][ch] = red

    # %% Plot
    # Selecionando Canal 1 para o plot principal, conforme o código original focava
    ch_to_plot = 1 
    
    devices_plot_idx = range(len(Devices)) # Plotar todos os pontos

    # FIGURA 1: Normalized Throughput
    plt.figure(1, figsize=(8, 6))
    
    plt.plot(Devices, res_data['QL'][ch_to_plot]['ntput'], 'o-', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL-NOMA')
    plt.plot(Devices, res_data['SA'][ch_to_plot]['ntput'], '*-', linewidth=1.5, markersize=8, label='SA-NOMA')
    
    # Reset color cycle logic simulation manually if needed, but default colors are fine
    plt.plot(Devices, res_data['QL_NoNOMA'][ch_to_plot]['ntput'], 'o:', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL without NOMA')
    plt.plot(Devices, res_data['SA_NoNOMA'][ch_to_plot]['ntput'], '*:', linewidth=1.5, markersize=8, label='SA without NOMA')

    plt.grid(True)
    plt.xlabel('Number of Devices (D)', fontsize=14)
    plt.ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f'Throughput ({ch_to_plot} Channel)', fontsize=14)
    plt.tight_layout()

    # Save the plot to a file instead of trying to open a GUI window
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Devices_fig1.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")

    # FIGURA 2: Rate of Redundant Messages
    plt.figure(2, figsize=(8, 6))

    plt.plot(Devices, redundancy['QL'][ch_to_plot], 'o-', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL-NOMA')
    plt.plot(Devices, redundancy['SA'][ch_to_plot], '*-', linewidth=1.5, markersize=8, label='SA-NOMA')
    plt.plot(Devices, redundancy['QL_NoNOMA'][ch_to_plot], 'o:', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL without NOMA')
    plt.plot(Devices, redundancy['SA_NoNOMA'][ch_to_plot], '*:', linewidth=1.5, markersize=8, label='SA without NOMA')

    plt.grid(True)
    plt.xlabel('Number of Devices (D)', fontsize=14)
    plt.ylabel(r'Rate of Redundant Messages ($\rho$) [%]', fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f'Redundancy ({ch_to_plot} Channel)', fontsize=14)
    plt.tight_layout()

    # Save the plot to a file instead of trying to open a GUI window
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Devices_fig2.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")

if __name__ == "__main__":
    run_simulation()