import matplotlib                                   # Importa Matplotlib para configuração do backend
matplotlib.use('Agg')                               # Define backend não-interativo (sem janela gráfica)
import numpy as np                                   # Importa NumPy para operações numéricas com arrays
import matplotlib.pyplot as plt                      # Importa interface de plotagem do Matplotlib
from datetime import datetime                        # Importa datetime para timestamps nas mensagens de log
import os                                            # Importa os para manipulação de caminhos e processos
from concurrent.futures import ProcessPoolExecutor, as_completed  # Importa executor de processos paralelos

from StochasticGeometry import StochasticGeometry    # Importa gerador de geometria estocástica (posições aleatórias)
from SlottedAloha import SlottedAloha_MultipleChannels  # Importa simulador Slotted Aloha com NOMA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_UniqueChannel  # Importa funções Q-Learning


# ==============================================================================
# WORKER — executa uma iteração de potência em seu próprio processo
# ==============================================================================
def _run_power_iteration(args):
    """
    Função worker que simula uma única potência de transmissão.
    Executada em processo separado pelo ProcessPoolExecutor.

    Args:
        args (tuple): Tupla com todos os parâmetros necessários para a simulação.

    Returns:
        tuple: (índice_potência, resultados_SA, resultados_QL, resultados_QL_UC).
    """
    # Desempacota todos os parâmetros recebidos via tupla
    (d, P_val, P_dBm_val, Devices, Relays, Channels_Relays,
     runs, slots, frames, alpha, gamma, r, N, Distance, h_Nakagami) = args

    # Exibe mensagem de progresso com timestamp e PID do processo
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Power: {P_dBm_val} dBm")

    # Calcula atenuação de percurso (path loss) usando modelo COST-231: L = 128.1 + 36.7*log10(d)
    alpha_k_j = 10 ** (-(128.1 + 36.7 * np.log10(Distance)) / 10)
    # Calcula SNR recebida: potência transmitida / ruído * atenuação * ganho do canal Nakagami
    SNR = P_val / N * alpha_k_j * h_Nakagami ** 2

    sa_mc = {}    # Dicionário para resultados SA-NOMA por configuração de canais
    ql_mc = {}    # Dicionário para resultados QL-NOMA por configuração de canais
    ql_uc = {}    # Dicionário para resultados QL-UniqueChannel por configuração de canais

    # Itera sobre cada configuração de número de canais
    for i, ch in enumerate(Channels_Relays):
        # --- Slotted Aloha com NOMA ---
        tp, dist, total = SlottedAloha_MultipleChannels(
            Devices, Relays, ch, runs, frames, slots, SNR, N, r)       # Executa simulação SA-NOMA
        sa_mc[i] = (tp, dist, total)                                    # Armazena throughput, tráfego distinto e total

        # --- Q-Learning com NOMA ---
        QTable = InitializeQTable(Devices, ch, slots, runs, True)       # Inicializa Q-Table com zeros
        tp, dist, total = Qlearning_MultipleChannels(
            Devices, Relays, ch, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)  # Executa QL-NOMA
        ql_mc[i] = (tp, dist, total)                                    # Armazena resultados QL-NOMA

        # --- Q-Learning com Canal Único (Ortogonal) ---
        QTable = InitializeQTable(Devices, ch, slots, runs, True)       # Reinicializa Q-Table para nova simulação
        tp, dist, total = Qlearning_UniqueChannel(
            Devices, Relays, ch, runs, frames, slots, SNR, N, r, QTable, alpha, gamma)  # Executa QL-UniqueChannel
        ql_uc[i] = (tp, dist, total)                                    # Armazena resultados QL-UniqueChannel

    return d, sa_mc, ql_mc, ql_uc  # Retorna índice de potência e todos os resultados


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_simulation():
    """
    Função principal que configura os parâmetros, executa as simulações em paralelo
    e gera os gráficos de throughput vs potência de transmissão.
    """
    # Configurações dos parâmetros da simulação
    np.random.seed(0)                                # Fixa semente aleatória para reprodutibilidade

    # --- Parâmetros dos Dispositivos ---
    Devices = 1500                                   # Número total de dispositivos IoT na rede

    # --- Faixa de Potência de Transmissão ---
    P_dBm = np.arange(-20, 35, 5)                   # Potências em dBm: de -20 a 30 com passo 5
    P_range = 10 ** ((P_dBm - 30) / 10)             # Converte dBm para Watts lineares

    # --- Parâmetros de Ruído ---
    FdB = 6                                          # Figura de ruído do receptor em dB
    F = 10 ** (FdB / 10)                             # Converte figura de ruído para linear
    N0dB = -204                                      # Densidade espectral de potência do ruído térmico (dBW/Hz)
    N0 = 10 ** (N0dB / 10)                           # Converte para Watts/Hz
    B = 100e3                                        # Largura de banda do canal em Hz (100 kHz)
    N = N0 * B * F                                   # Potência total do ruído: N0 * B * F

    # --- Parâmetros de Relays e Canais ---
    Relays = 3                                       # Número de relays (receptores)
    Channels_Relays = [1, 2, 3]                      # Configurações de canais a simular

    r = 3                                            # Taxa alvo mínima (bps/Hz) para decodificação

    # --- Parâmetros de Satélite ---
    altitude = 780e3                                 # Altitude do satélite em metros (780 km)

    # --- Parâmetros de Geometria Estocástica ---
    cell_radius = 5e3                                # Raio da célula em metros (5 km)

    # --- Parâmetros de Simulação ---
    runs = 100                                       # Número de rodadas Monte Carlo
    slots = 100                                      # Número de slots de tempo por quadro
    frames = 50                                      # Número de quadros de transmissão

    alpha = 0.1                                      # Taxa de aprendizado do Q-Learning
    gamma = 0.5                                      # Fator de desconto do Q-Learning

    # --- Canal Nakagami-m ---
    m = 2                                            # Parâmetro de forma Nakagami (m=2: desvanecimento moderado)

    # --- Geometria Estocástica: gera posições aleatórias e calcula distâncias ---
    Distance = StochasticGeometry(Devices, Relays, cell_radius, runs)  # Matriz de distâncias (Devices x Relays x runs)

    # --- Geração do canal Nakagami complexo ---
    shape = m / 2                                    # Parâmetro shape da distribuição Gamma
    scale = 1                                        # Parâmetro scale da distribuição Gamma
    size = (Devices, Relays, runs)                   # Dimensões da matriz do canal

    h_nak_real = np.sqrt(np.random.gamma(shape, scale, size))    # Parte real do canal Nakagami
    h_nak_imag = np.sqrt(np.random.gamma(shape, scale, size))    # Parte imaginária do canal Nakagami
    h_Nakagami = np.abs((h_nak_real + 1j * h_nak_imag) / np.sqrt(m))  # Módulo do canal normalizado

    n_p = len(P_dBm)                                # Número de níveis de potência a simular

    # --- Pré-alocação das estruturas de resultados ---
    # SA-NOMA: throughput, tráfego distinto e total para cada configuração de canais
    NormThroughput_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}      # Throughput normalizado SA-NOMA
    ndist_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}              # Tráfego distinto SA-NOMA
    ntotal_SA_MC = {ch: np.zeros(n_p) for ch in range(3)}             # Tráfego total SA-NOMA

    # QL-NOMA: throughput, tráfego distinto e total
    NormThroughput_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}      # Throughput normalizado QL-NOMA
    ndist_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}              # Tráfego distinto QL-NOMA
    ntotal_QL_MC = {ch: np.zeros(n_p) for ch in range(3)}             # Tráfego total QL-NOMA

    # QL-UniqueChannel: throughput, tráfego distinto e total
    NormThroughput_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}      # Throughput normalizado QL-UC
    ndist_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}              # Tráfego distinto QL-UC
    ntotal_QL_UC = {ch: np.zeros(n_p) for ch in range(3)}             # Tráfego total QL-UC

    # Mensagem de início da simulação
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando simulação...")

    # Monta argumentos para cada nível de potência (um bundle por worker)
    task_args = [
        (d, P_range[d], P_dBm[d], Devices, Relays, Channels_Relays,
         runs, slots, frames, alpha, gamma, r, N, Distance, h_Nakagami)
        for d in range(n_p)
    ]

    # Determina número de workers: mínimo entre CPUs disponíveis e níveis de potência
    n_workers = min(os.cpu_count() or 1, n_p)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_p} power levels.")

    # --- Execução paralela: distribui cada nível de potência para um processo ---
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_power_iteration, args): args[0] for args in task_args}  # Submete tarefas
        for future in as_completed(futures):                                                     # Coleta resultados conforme concluem
            d, sa_mc, ql_mc, ql_uc = future.result()                                            # Desempacota resultado
            for i in range(len(Channels_Relays)):                                                # Armazena resultados por canal
                NormThroughput_SA_MC[i][d], ndist_SA_MC[i][d], ntotal_SA_MC[i][d] = sa_mc[i]    # Resultados SA-NOMA
                NormThroughput_QL_MC[i][d], ndist_QL_MC[i][d], ntotal_QL_MC[i][d] = ql_mc[i]    # Resultados QL-NOMA
                NormThroughput_QL_UC[i][d], ndist_QL_UC[i][d], ntotal_QL_UC[i][d] = ql_uc[i]    # Resultados QL-UC

    # --- Cálculo das taxas de redundância: 100*(1 - distinto/total) ---
    redundant_SA_MC = {}                             # Taxa de redundância SA-NOMA
    redundant_QL_MC = {}                             # Taxa de redundância QL-NOMA
    redundant_QL_UC = {}                             # Taxa de redundância QL-UC
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):                                     # Ignora warnings de divisão por zero
            redundant_SA_MC[i] = np.nan_to_num(100 * (1 - ndist_SA_MC[i] / ntotal_SA_MC[i]))    # Redundância SA-NOMA (NaN→0)
            redundant_QL_MC[i] = np.nan_to_num(100 * (1 - ndist_QL_MC[i] / ntotal_QL_MC[i]))    # Redundância QL-NOMA (NaN→0)
            redundant_QL_UC[i] = np.nan_to_num(100 * (1 - ndist_QL_UC[i] / ntotal_QL_UC[i]))    # Redundância QL-UC (NaN→0)

    # --- Cores padrão do MATLAB para consistência visual ---
    c1 = [0, 0.4470, 0.7410]                        # Azul MATLAB
    c2 = [0.8500, 0.3250, 0.0980]                   # Laranja MATLAB
    c3 = [0.9290, 0.6940, 0.1250]                   # Amarelo MATLAB

    P_plot = list(range(len(P_dBm)))                 # Índices para posicionamento dos marcadores

    # --- Figura 1: Taxa de Redundância vs Potência ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))                                                           # Cria figura 10x6 polegadas
    ax1.plot(P_dBm, redundant_SA_MC[0], 's-', linewidth=1.5, color=c1, label='SA-NOMA - C=1')           # SA-NOMA com 1 canal
    ax1.plot(P_dBm, redundant_QL_MC[0], 's--', linewidth=1.5, color=c1, label='QL-NOMA - C=1')          # QL-NOMA com 1 canal
    ax1.plot(P_dBm, redundant_SA_MC[1], 'd-', linewidth=1.5, color=c2, label='SA-NOMA - C=2')           # SA-NOMA com 2 canais
    ax1.plot(P_dBm, redundant_QL_MC[1], 'd--', linewidth=1.5, color=c2, label='QL-NOMA - C=2')          # QL-NOMA com 2 canais
    ax1.plot(P_dBm, redundant_SA_MC[2], 'o-', linewidth=1.5, color=c3, label='SA-NOMA - C=3')           # SA-NOMA com 3 canais
    ax1.plot(P_dBm, redundant_QL_MC[2], 'o--', linewidth=1.5, color=c3, label='QL-NOMA - C=3')          # QL-NOMA com 3 canais
    ax1.grid(True)                                                                                        # Ativa grade no gráfico
    ax1.legend(fontsize=12)                                                                               # Exibe legenda
    ax1.set_xlabel('Power (dBm)', fontsize=14)                                                            # Rótulo do eixo X
    ax1.set_ylabel('Rate of redundant messages', fontsize=14)                                             # Rótulo do eixo Y

    # Salva figura 1 no diretório de resultados (relativo ao script)
    script_dir = os.path.dirname(os.path.abspath(__file__))                                              # Diretório do script atual
    output_path1 = os.path.join(script_dir, '..', 'results', 'Throughput_Power_fig1.png')                # Caminho de saída da figura 1
    fig1.savefig(output_path1, bbox_inches='tight', dpi=150)                                             # Salva figura em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path1}")             # Log de confirmação

    # --- Figura 2: Throughput Normalizado vs Potência ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))                                                           # Cria segunda figura
    ax2.plot(P_dBm, NormThroughput_QL_MC[0], 's-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='QL-NOMA (C=1)')                                  # QL-NOMA com 1 canal
    ax2.plot(P_dBm, NormThroughput_SA_MC[0], 's:', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='SA-NOMA (C=1)')                                  # SA-NOMA com 1 canal
    ax2.set_prop_cycle(None)                                                                              # Reseta ciclo de cores
    ax2.plot(P_dBm, NormThroughput_QL_MC[2], 'o-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='QL-NOMA (C=3)')                                  # QL-NOMA com 3 canais
    ax2.plot(P_dBm, NormThroughput_SA_MC[2], 'o:', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='SA-NOMA (C=3)')                                  # SA-NOMA com 3 canais
    ax2.plot(P_dBm, NormThroughput_QL_UC[2], '*-', linewidth=1.5, markevery=P_plot,
             markerfacecolor='w', markersize=10, label='DQL-based JRSAC [4] (C=3)')                      # QL-UniqueChannel com 3 canais
    ax2.grid(True)                                                                                        # Ativa grade
    ax2.legend(fontsize=12)                                                                               # Exibe legenda
    ax2.set_xlabel('Transmit Power (P) [dBm]', fontsize=14)                                              # Rótulo eixo X
    ax2.set_ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)                               # Rótulo eixo Y com LaTeX

    # Salva figura 2 no diretório de resultados
    output_path2 = os.path.join(script_dir, '..', 'results', 'Throughput_Power_fig2.png')                # Caminho de saída da figura 2
    fig2.savefig(output_path2, bbox_inches='tight', dpi=150)                                             # Salva figura em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path2}")             # Log de confirmação

# Ponto de entrada: executa simulação quando o script é chamado diretamente
if __name__ == "__main__":
    run_simulation()                                 # Inicia a simulação principal