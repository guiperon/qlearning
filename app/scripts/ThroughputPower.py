import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from StochasticGeometry import StochasticGeometry
from SlottedAloha import SlottedAloha_MultipleChannels
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_UniqueChannel

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def run_simulation():
    # Parameters
    np.random.seed(0)

    # Devices
    Devices = 1500

    # Power range
    P_dBm = np.arange(-20, 35, 5)  # -20:5:30
    P_range = 10 ** ((P_dBm - 30) / 10)

    FdB = 6
    F = 10 ** (FdB / 10)
    N0dB = -204
    N0 = 10 ** (N0dB / 10)
    B = 100e3
    N = N0 * B * F

    # Relays
    Relays = 3
    Channels_Relays = [1, 2, 3]

    r = 3

    # Satellites
    altitude = 780e3

    # Stochastic Geometry Parameters
    cell_radius = 5e3

    # Simulation Parameters
    runs = 100
    slots = 100
    frames = 50

    alpha = 0.1
    gamma = 0.5

    # Nakagami-m
    m = 2

    # Stochastic Geometry
    Distance = StochasticGeometry(Devices, Relays, cell_radius, runs)

    shape = m / 2
    scale = 1
    size = (Devices, Relays, runs)

    h_nak_real = np.sqrt(np.random.gamma(shape, scale, size))
    h_nak_imag = np.sqrt(np.random.gamma(shape, scale, size))
    h_Nakagami = np.abs((h_nak_real + 1j * h_nak_imag) / np.sqrt(m))

    n_p = len(P_dBm)

    # SA-NOMA results
    NormThroughput_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}
    ndist_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}
    ntotal_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}

    # QL-NOMA results
    NormThroughput_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}
    ndist_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}
    ntotal_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}

    # QL UniqueChannel results
    NormThroughput_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}
    ndist_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}
    ntotal_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando simulação...")

    for d in range(n_p):
        P = P_range[d]
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Power: {P_dBm[d]}/{P_dBm[-1]} dBm")

        alpha_k_j = 10 ** (-(128.1 + 36.7 * np.log10(Distance)) / 10)
        SNR = P / N * alpha_k_j * h_Nakagami ** 2

        for i, ch in enumerate(Channels_Relays):
            # SA-NOMA
            tp, dist, total = SlottedAloha_MultipleChannels(
                Devices, Relays, ch, runs, frames, slots, SNR, N, r)
            NormThroughput_SA_MC[i][d] = tp
            ndist_SA_MC[i][d] = dist
            ntotal_SA_MC[i][d] = total

            # QL-NOMA
            QTable = InitializeQTable(Devices, ch, slots, runs, True)
            tp, dist, total = Qlearning_MultipleChannels(
                Devices, Relays, ch, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)
            NormThroughput_QL_MC[i][d] = tp
            ndist_QL_MC[i][d] = dist
            ntotal_QL_MC[i][d] = total

            # QL UniqueChannel
            QTable = InitializeQTable(Devices, ch, slots, runs, True)
            tp, dist, total = Qlearning_UniqueChannel(
                Devices, Relays, ch, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)
            NormThroughput_QL_UC[i][d] = tp
            ndist_QL_UC[i][d] = dist
            ntotal_QL_UC[i][d] = total

    # Redundant rates
    redundant_SA_MC = {}
    redundant_QL_MC = {}
    redundant_QL_UC = {}
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            redundant_SA_MC[i] = np.nan_to_num(100 * (1 - ndist_SA_MC[i] / ntotal_SA_MC[i]))
            redundant_QL_MC[i] = np.nan_to_num(100 * (1 - ndist_QL_MC[i] / ntotal_QL_MC[i]))
            redundant_QL_UC[i] = np.nan_to_num(100 * (1 - ndist_QL_UC[i] / ntotal_QL_UC[i]))

    # MATLAB default colors
    c1 = [0, 0.4470, 0.7410]
    c2 = [0.8500, 0.3250, 0.0980]
    c3 = [0.9290, 0.6940, 0.1250]

    P_plot = list(range(len(P_dBm)))

    # Figure 1 - Redundant Rate
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(P_dBm, redundant_SA_MC[0], 's-', linewidth=1.5, color=c1, label='SA-NOMA - C=1')
    ax1.plot(P_dBm, redundant_QL_MC[0], 's--', linewidth=1.5, color=c1, label='QL-NOMA - C=1')
    ax1.plot(P_dBm, redundant_SA_MC[1], 'd-', linewidth=1.5, color=c2, label='SA-NOMA - C=2')
    ax1.plot(P_dBm, redundant_QL_MC[1], 'd--', linewidth=1.5, color=c2, label='QL-NOMA - C=2')
    ax1.plot(P_dBm, redundant_SA_MC[2], 'o-', linewidth=1.5, color=c3, label='SA-NOMA - C=3')
    ax1.plot(P_dBm, redundant_QL_MC[2], 'o--', linewidth=1.5, color=c3, label='QL-NOMA - C=3')
    ax1.grid(True)
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Power (dBm)', fontsize=14)
    ax1.set_ylabel('Rate of redundant messages', fontsize=14)

    output_path1 = '/workspaces/testepy/qlearning0/Throughput_Power_fig1.png'
    fig1.savefig(output_path1, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path1}")

    # Figure 2 - Normalized Throughput
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(P_dBm, NormThroughput_QL_MC[0], 's-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='QL-NOMA (C=1)')
    ax2.plot(P_dBm, NormThroughput_SA_MC[0], 's:', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='SA-NOMA (C=1)')
    ax2.set_prop_cycle(None)  # Reset color cycle
    ax2.plot(P_dBm, NormThroughput_QL_MC[2], 'o-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='QL-NOMA (C=3)')
    ax2.plot(P_dBm, NormThroughput_SA_MC[2], 'o:', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='SA-NOMA (C=3)')
    ax2.plot(P_dBm, NormThroughput_QL_UC[2], '*-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='DQL-based JRSAC [4] (C=3)')
    ax2.grid(True)
    ax2.legend(fontsize=12)
    ax2.set_xlabel('Transmit Power (P) [dBm]', fontsize=14)
    ax2.set_ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)

    output_path2 = '/workspaces/testepy/qlearning0/Throughput_Power_fig2.png'
    fig2.savefig(output_path2, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path2}")

if __name__ == "__main__":
    run_simulation()