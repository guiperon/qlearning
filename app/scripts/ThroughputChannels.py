import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from StochasticGeometry import StochasticGeometry
from SlottedAloha import SlottedAloha_MultipleChannels, SlottedAloha_MultipleChannels_NoNOMA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_MultipleChannels_NoNOMA


def _run_channel_iteration(args):
    (idx, num_channels, Devices, Relays, runs, frames, slots,
     N, r, alpha, gamma, P, Distance, h_Nakagami) = args

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Channels: {num_channels}")

    alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(Distance)) / 10)
    SNR = (P / N * alpha_k_j * h_Nakagami**2)

    tp_sa, ndist_sa, ntotal_sa = SlottedAloha_MultipleChannels(
        Devices, Relays, num_channels, runs, frames, slots, SNR, N, r
    )

    QTable = InitializeQTable(Devices, num_channels, slots, runs, True)
    tp_ql, ndist_ql, ntotal_ql = Qlearning_MultipleChannels(
        Devices, Relays, num_channels, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
    )

    return idx, tp_sa, ndist_sa, ntotal_sa, tp_ql, ndist_ql, ntotal_ql

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def run_simulation():
    # %% Parameters
    np.random.seed(0)

    # Devices (Fixed number)
    Devices = 1500

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
    
    # Variando de 1 a 10 canais
    Channels_Relays = np.arange(1, 11)

    r = 3

    # Stochastic Geometry Parameters
    cell_radius = 5e3

    # Satellites
    altitude = 780e3

    # Simulation Parameters
    runs = 100    # original: 100
    slots = 100   # original: 100
    frames = 50   # original: 50

    alpha = 0.1
    gamma = 0.5

    # Nakagami-m
    m = 2

    # %% Stochastic Geometry
    # Calcula geometria e canal uma única vez (pois Devices é fixo)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gerando Geometria Estocástica...")
    Distance = StochasticGeometry(Devices, Relays, cell_radius, runs)
    
    # Geração Nakagami complexa: sqrt(Gamma) + j*sqrt(Gamma)
    shape_k = m / 2
    scale_theta = 1
    size_h = (Devices, Relays, runs)
    
    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))
    
    n_cases = len(Channels_Relays)

    # Resultados indexados por quantidade de canais
    results_sa = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}
    results_ql = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando loop de simulação...")

    task_args = [
        (idx, int(num_channels), Devices, Relays, runs, frames, slots,
         N, r, alpha, gamma, P, Distance, h_Nakagami)
        for idx, num_channels in enumerate(Channels_Relays)
    ]

    n_workers = min(os.cpu_count() or 1, n_cases)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_cases} channel cases.")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_channel_iteration, args): args[0] for args in task_args}
        for future in as_completed(futures):
            idx, tp_sa, ndist_sa, ntotal_sa, tp_ql, ndist_ql, ntotal_ql = future.result()
            results_sa['ntput'][idx] = tp_sa
            results_sa['ndist'][idx] = ndist_sa
            results_sa['ntotal'][idx] = ntotal_sa
            results_ql['ntput'][idx] = tp_ql
            results_ql['ndist'][idx] = ndist_ql
            results_ql['ntotal'][idx] = ntotal_ql

    # Conversão para arrays numpy para facilitar cálculos
    ndist_sa_arr = np.array(results_sa['ndist'])
    ntotal_sa_arr = np.array(results_sa['ntotal'])

    ndist_ql_arr = np.array(results_ql['ndist'])
    ntotal_ql_arr = np.array(results_ql['ntotal'])

    # Cálculo da Redundância
    # MATLAB: 100*(1 - ndist./ntotal)
    with np.errstate(divide='ignore', invalid='ignore'):
        redundantRate_SA = 100 * (1 - ndist_sa_arr / ntotal_sa_arr)
        redundantRate_QL = 100 * (1 - ndist_ql_arr / ntotal_ql_arr)
    
    # Limpeza de NaNs (caso ntotal seja 0)
    redundantRate_SA = np.nan_to_num(redundantRate_SA)
    redundantRate_QL = np.nan_to_num(redundantRate_QL)

    # %% Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo da Esquerda (Throughput) - Linhas Sólidas
    # Truque para a legenda aparecer corretamente com os estilos dos dois eixos
    line1, = ax1.plot(Channels_Relays, results_ql['ntput'], 'o-', color='tab:blue', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL-NOMA (Throughput)')
    line2, = ax1.plot(Channels_Relays, results_sa['ntput'], '*-', color='tab:orange', linewidth=1.5, markersize=8, label='SA-NOMA (Throughput)')
    
    ax1.set_xlabel('Number of Channels (C)', fontsize=14)
    ax1.set_ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)
    ax1.grid(True)

    # Eixo da Direita (Redundancy) - Linhas Pontilhadas
    ax2 = ax1.twinx()
    line3, = ax2.plot(Channels_Relays, redundantRate_QL, 'o:', color='tab:blue', linewidth=1.5, markersize=8, markerfacecolor='w', label='QL-NOMA (Redundancy)')
    line4, = ax2.plot(Channels_Relays, redundantRate_SA, '*:', color='tab:orange', linewidth=1.5, markersize=8, label='SA-NOMA (Redundancy)')
    
    ax2.set_ylabel(r'Rate of Redundant Messages ($\rho$) [%]', fontsize=14)

    # Legenda Unificada
    # lines = [line1, line2, line3, line4]
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='best', fontsize=12)
    
    # Ou legenda simplificada conforme o original
    plt.title("Impact of Number of Channels on Throughput and Redundancy", fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88), fontsize=10)

    plt.tight_layout()

    # Save the plot to a file instead of trying to open a GUI window
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Channels_fig1.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")

if __name__ == "__main__":
    run_simulation()