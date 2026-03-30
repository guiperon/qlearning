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
    # %% Parameters
    np.random.seed(0)

    # Devices
    Devices = 1500

    P_dBm = 10
    P = 10**((P_dBm - 30) / 10)

    FdB = 6
    F = 10**(FdB / 10)
    N0dB = -204
    N0 = 10**(N0dB / 10)
    B = 100e3
    N = N0 * B * F

    # Range de Relays/Channels (1 a 15)
    Relays_Range = np.arange(1, 16)
    Channels_Relays_Range = Relays_Range # R = C

    r = 3

    # Stochastic Geometry Parameters
    cell_radius = 5e3

    # Satellites
    altitude = 780e3

    # Simulation Parameters
    runs = 100
    slots = 100
    frames = 10 # Original: 10

    alpha = 0.1
    gamma = 0.5

    # Nakagami-m
    m = 2

    # %% Stochastic Geometry
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gerando Geometria Estocástica...")
    
    max_relays = np.max(Relays_Range)
    
    Distance = StochasticGeometry(Devices, max_relays, cell_radius, runs)
    
    # Canal Nakagami
    shape_k = m / 2
    scale_theta = 1
    size_h = (Devices, max_relays, runs)
    
    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))

    # Estruturas de resultados
    res_sa = {'ntput': [], 'ndist': [], 'ntotal': []}
    res_ql = {'ntput': [], 'ndist': [], 'ntotal': []}
    res_ql_unique = {'ntput': [], 'ndist': [], 'ntotal': []}

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Loop de Simulação...")

    for d_idx, n_relays in enumerate(Relays_Range):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Number of Relays: {n_relays}/{max_relays}")
        
        Channels_Relays = Channels_Relays_Range[d_idx]

        # Path Loss e SNR
        # Nota: Distance tem dimensões (Devices, max_relays, runs).
        # Precisamos usar apenas as colunas correspondentes ao número atual de relés?
        # O código original usa "Distance" diretamente no log10, o que implica que ele usa
        # a matriz inteira ou a dimensão bate.
        # Como 'h_Nakagami' e 'Distance' foram gerados com 'max(Relays_Range)', 
        # para a simulação atual com 'n_relays', deveríamos pegar um slice.
        # O código MATLAB original passava SNR completo para as funções, mas dentro delas
        # ele itera até 'Relays'. Vamos passar o slice correto para evitar erros de dimensão.
        
        dist_slice = Distance[:, :n_relays, :]
        h_slice = h_Nakagami[:, :n_relays, :]
        
        alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(dist_slice)) / 10)
        SNR = (P / N * alpha_k_j * h_slice**2)

        # SA-NOMA
        tp, nd, nt = SlottedAloha_MultipleChannels(Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r)
        res_sa['ntput'].append(tp)
        res_sa['ndist'].append(nd)
        res_sa['ntotal'].append(nt)
        
        # QL-NOMA
        QTable = InitializeQTable(Devices, Channels_Relays, slots, runs, True)
        tp, nd, nt = Qlearning_MultipleChannels(Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)
        res_ql['ntput'].append(tp)
        res_ql['ndist'].append(nd)
        res_ql['ntotal'].append(nt)

        # QL-Unique (Orthogonal)
        QTable = InitializeQTable(Devices, Channels_Relays, slots, runs, True)
        tp, nd, nt = Qlearning_UniqueChannel(Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)
        res_ql_unique['ntput'].append(tp)
        res_ql_unique['ndist'].append(nd)
        res_ql_unique['ntotal'].append(nt)

    # Cálculo das taxas de redundância
    def calc_red(r_dict):
        nd = np.array(r_dict['ndist'])
        nt = np.array(r_dict['ntotal'])
        with np.errstate(divide='ignore', invalid='ignore'):
            red = 100 * (1 - nd / nt)
        return np.nan_to_num(red)

    red_sa = calc_red(res_sa)
    red_ql = calc_red(res_ql)
    red_ql_unique = calc_red(res_ql_unique)

    # %% Plotagem
    # Figura 1: Bar Chart (Redundância)
    plt.figure(1, figsize=(10, 6))
    
    # Prepara dados para barra agrupada
    # x = Relays_Range
    width = 0.25
    x = np.arange(len(Relays_Range))
    
    plt.bar(x - width, red_sa, width, label='SA-NOMA', color='#0072BD') # Azul
    plt.bar(x, red_ql, width, label='QL-NOMA', color='#D95319')        # Laranja
    plt.bar(x + width, red_ql_unique, width, label='QL Orthogonal', color='#EDB120') # Amarelo

    plt.xlabel('Number of Relays = Number of Channels', fontsize=13)
    plt.ylabel('Rate of Redundant Messages (%)', fontsize=13)
    plt.legend()
    plt.grid(True, axis='y')
    plt.xticks(x, Relays_Range) # Labels corretos no eixo X
    plt.xlim([-0.5, len(Relays_Range) - 0.5])
    plt.title('Redundancy Rate Comparison')
    plt.tight_layout()

    # Figura 2: Line Plot (Throughput)
    plt.figure(2, figsize=(10, 6))
    
    plt.plot(Relays_Range, res_ql['ntput'], 'o-', linewidth=1.5, markersize=10, markerfacecolor='w', label='QL-NOMA')
    plt.plot(Relays_Range, res_ql_unique['ntput'], 's-', linewidth=1.5, markersize=10, markerfacecolor='w', label='DQL-based JRSAC [4]')
    plt.plot(Relays_Range, res_sa['ntput'], '*-', linewidth=1.5, markersize=10, markerfacecolor='w', label='SA-NOMA')

    plt.grid(True)
    plt.xlabel('Number of Relays = Number of Channels (R = C)', fontsize=14)
    plt.ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Normalized Throughput Comparison')
    plt.tight_layout()

    # Save the plot to a file instead of trying to open a GUI window
    output_path = '/workspaces/testepy/qlearning0/Throughput_Channels_Relays_fig1.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")

if __name__ == "__main__":
    run_simulation()