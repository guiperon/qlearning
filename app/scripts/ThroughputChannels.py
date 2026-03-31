import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from StochasticGeometry import StochasticGeometry
from SlottedAloha import SlottedAloha_MultipleChannels, SlottedAloha_MultipleChannels_NoNOMA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_MultipleChannels_NoNOMA

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
    
    # Listas para armazenar resultados
    results_sa = {'ntput': [], 'ndist': [], 'ntotal': []}
    results_ql = {'ntput': [], 'ndist': [], 'ntotal': []}

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando loop de simulação...")

    # Loop variando o número de canais disponíveis
    for d_idx, num_channels in enumerate(Channels_Relays):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Channels: {num_channels}/{np.max(Channels_Relays)}")
        
        # Path Loss e SNR
        # Nota: np.log10
        alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(Distance)) / 10)
        SNR = (P / N * alpha_k_j * h_Nakagami**2)

        # --- SA-NOMA ---
        tp_sa, ndist_sa, ntotal_sa = SlottedAloha_MultipleChannels(
            Devices, Relays, num_channels, runs, frames, slots, SNR, N, r
        )
        results_sa['ntput'].append(tp_sa)
        results_sa['ndist'].append(ndist_sa)
        results_sa['ntotal'].append(ntotal_sa)

        # --- Q-Learning ---
        # Inicializa a tabela com o número atual de canais
        QTable = InitializeQTable(Devices, num_channels, slots, runs, True)
        
        tp_ql, ndist_ql, ntotal_ql = Qlearning_MultipleChannels(
            Devices, Relays, num_channels, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
        )
        results_ql['ntput'].append(tp_ql)
        results_ql['ndist'].append(ndist_ql)
        results_ql['ntotal'].append(ntotal_ql)

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
    output_path = '../results/Throughput_Channels_fig1.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")

if __name__ == "__main__":
    run_simulation()