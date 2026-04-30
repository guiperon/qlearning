import numpy as np                                   # Importa NumPy para operações numéricas com arrays
import matplotlib.pyplot as plt                      # Importa interface de plotagem do Matplotlib
from datetime import datetime                        # Importa datetime para timestamps nas mensagens de log
import os                                            # Importa os para manipulação de caminhos e processos
from concurrent.futures import ProcessPoolExecutor, as_completed  # Importa executor de processos paralelos

from StochasticGeometry import StochasticGeometry    # Importa gerador de geometria estocástica
from SlottedAloha import SlottedAloha_MultipleChannels, SlottedAloha_MultipleChannels_NoNOMA  # Importa simuladores SA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_MultipleChannels_NoNOMA  # Importa funções QL

# ==============================================================================
# WORKER — executa uma iteração de canais em seu próprio processo
# ==============================================================================
def _run_channel_iteration(args):
    """
    Função worker que simula uma única configuração de número de canais.
    Executada em processo separado pelo ProcessPoolExecutor.

    Args:
        args (tuple): Tupla com todos os parâmetros necessários.

    Returns:
        tuple: (índice, throughput_SA, distinto_SA, total_SA, throughput_QL, distinto_QL, total_QL).
    """
    # Desempacota parâmetros recebidos
    (idx, num_channels, Devices, Relays, runs, frames, slots,
     N, r, alpha, gamma, P, Distance, h_Nakagami) = args

    # Exibe mensagem de progresso com timestamp e PID do processo
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Channels: {num_channels}")

    # Calcula atenuação de percurso (path loss) usando modelo COST-231
    alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(Distance)) / 10)
    # Calcula SNR recebida: potência / ruído * atenuação * ganho Nakagami
    SNR = (P / N * alpha_k_j * h_Nakagami**2)

    # Executa simulação Slotted Aloha com NOMA
    tp_sa, ndist_sa, ntotal_sa = SlottedAloha_MultipleChannels(
        Devices, Relays, num_channels, runs, frames, slots, SNR, N, r
    )

    # Inicializa Q-Table com zeros e executa Q-Learning com NOMA
    QTable = InitializeQTable(Devices, num_channels, slots, runs, True)
    tp_ql, ndist_ql, ntotal_ql = Qlearning_MultipleChannels(
        Devices, Relays, num_channels, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
    )

    return idx, tp_sa, ndist_sa, ntotal_sa, tp_ql, ndist_ql, ntotal_ql  # Retorna índice e resultados

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_simulation():
    """
    Função principal que configura parâmetros, executa simulações em paralelo
    e gera o gráfico de throughput e redundância vs número de canais.
    """
    # Configurações dos parâmetros
    np.random.seed(0)                                # Fixa semente aleatória para reprodutibilidade

    # --- Parâmetros dos Dispositivos ---
    Devices = 1500                                   # Número fixo de dispositivos IoT

    # --- Potência de Transmissão ---
    P_dBm = 10                                       # Potência de transmissão fixa em dBm
    P = 10**((P_dBm - 30) / 10)                     # Converte dBm para Watts lineares

    # --- Parâmetros de Ruído ---
    FdB = 6                                          # Figura de ruído do receptor (dB)
    F = 10**(FdB / 10)                               # Figura de ruído em linear
    N0dB = -204                                      # Densidade espectral de ruído térmico (dBW/Hz)
    N0 = 10**(N0dB / 10)                             # Densidade de ruído em Watts/Hz
    B = 100e3                                        # Largura de banda do canal (100 kHz)
    N = N0 * B * F                                   # Potência total do ruído

    # --- Parâmetros de Relays e Canais ---
    Relays = 4                                       # Número fixo de relays
    Channels_Relays = np.arange(1, 11)               # Varia de 1 a 10 canais (variável principal)

    r = 3                                            # Taxa alvo mínima (bps/Hz)

    # --- Parâmetros de Geometria Estocástica ---
    cell_radius = 5e3                                # Raio da célula (5 km)

    # --- Parâmetros de Satélite ---
    altitude = 780e3                                 # Altitude do satélite (780 km)

    # --- Parâmetros de Simulação ---
    runs = 100                                       # Rodadas Monte Carlo
    slots = 100                                      # Slots de tempo por quadro
    frames = 50                                      # Quadros de transmissão

    alpha = 0.1                                      # Taxa de aprendizado Q-Learning
    gamma = 0.5                                      # Fator de desconto Q-Learning

    # --- Canal Nakagami-m ---
    m = 2                                            # Parâmetro Nakagami (m=2: desvanecimento moderado)

    # --- Geometria Estocástica: gera posições e calcula distâncias ---
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gerando Geometria Estocástica...")
    Distance = StochasticGeometry(Devices, Relays, cell_radius, runs)   # Matriz (Devices x Relays x runs)

    # Geração do canal Nakagami complexo
    shape_k = m / 2                                  # Parâmetro shape da distribuição Gamma
    scale_theta = 1                                  # Parâmetro scale da distribuição Gamma
    size_h = (Devices, Relays, runs)                 # Dimensões da matriz do canal

    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte real do canal Nakagami
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte imaginária do canal Nakagami
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))           # Módulo do canal normalizado

    n_cases = len(Channels_Relays)                   # Número de configurações de canais a simular

    # --- Pré-alocação das estruturas de resultados ---
    results_sa = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}  # Resultados SA-NOMA
    results_ql = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}  # Resultados QL-NOMA

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando loop de simulação...")

    # Monta argumentos para cada configuração de canais
    task_args = [
        (idx, int(num_channels), Devices, Relays, runs, frames, slots,
         N, r, alpha, gamma, P, Distance, h_Nakagami)
        for idx, num_channels in enumerate(Channels_Relays)
    ]

    # Determina número de workers paralelos
    n_workers = min(os.cpu_count() or 1, n_cases)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_cases} channel cases.")

    # --- Execução paralela: distribui cada configuração de canais para um processo ---
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_channel_iteration, args): args[0] for args in task_args}  # Submete tarefas
        for future in as_completed(futures):                                                       # Coleta resultados
            idx, tp_sa, ndist_sa, ntotal_sa, tp_ql, ndist_ql, ntotal_ql = future.result()         # Desempacota
            results_sa['ntput'][idx] = tp_sa                                                       # Throughput SA-NOMA
            results_sa['ndist'][idx] = ndist_sa                                                    # Tráfego distinto SA
            results_sa['ntotal'][idx] = ntotal_sa                                                  # Tráfego total SA
            results_ql['ntput'][idx] = tp_ql                                                       # Throughput QL-NOMA
            results_ql['ndist'][idx] = ndist_ql                                                    # Tráfego distinto QL
            results_ql['ntotal'][idx] = ntotal_ql                                                  # Tráfego total QL

    # Converte para arrays numpy para cálculos vetorizados
    ndist_sa_arr = np.array(results_sa['ndist'])     # Array de tráfego distinto SA
    ntotal_sa_arr = np.array(results_sa['ntotal'])   # Array de tráfego total SA

    ndist_ql_arr = np.array(results_ql['ndist'])     # Array de tráfego distinto QL
    ntotal_ql_arr = np.array(results_ql['ntotal'])   # Array de tráfego total QL

    # --- Cálculo da taxa de redundância: 100*(1 - distinto/total) ---
    with np.errstate(divide='ignore', invalid='ignore'):                                           # Ignora warnings de divisão por zero
        redundantRate_SA = 100 * (1 - ndist_sa_arr / ntotal_sa_arr)                                # Redundância SA-NOMA
        redundantRate_QL = 100 * (1 - ndist_ql_arr / ntotal_ql_arr)                                # Redundância QL-NOMA

    redundantRate_SA = np.nan_to_num(redundantRate_SA)   # Substitui NaN por 0 (caso ntotal=0)
    redundantRate_QL = np.nan_to_num(redundantRate_QL)   # Substitui NaN por 0

    # --- Plotagem com eixo duplo (throughput + redundância) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))                                                      # Cria figura 10x6

    # Eixo esquerdo: Throughput (linhas sólidas)
    line1, = ax1.plot(Channels_Relays, results_ql['ntput'], 'o-', color='tab:blue', linewidth=1.5,
                      markersize=8, markerfacecolor='w', label='QL-NOMA (Throughput)')              # QL-NOMA throughput
    line2, = ax1.plot(Channels_Relays, results_sa['ntput'], '*-', color='tab:orange', linewidth=1.5,
                      markersize=8, label='SA-NOMA (Throughput)')                                   # SA-NOMA throughput

    ax1.set_xlabel('Number of Channels (C)', fontsize=14)                                          # Rótulo eixo X
    ax1.set_ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)                        # Rótulo eixo Y esquerdo
    ax1.grid(True)                                                                                  # Ativa grade

    # Eixo direito: Redundância (linhas pontilhadas)
    ax2 = ax1.twinx()                                                                              # Cria eixo Y compartilhado à direita
    line3, = ax2.plot(Channels_Relays, redundantRate_QL, 'o:', color='tab:blue', linewidth=1.5,
                      markersize=8, markerfacecolor='w', label='QL-NOMA (Redundancy)')              # QL-NOMA redundância
    line4, = ax2.plot(Channels_Relays, redundantRate_SA, '*:', color='tab:orange', linewidth=1.5,
                      markersize=8, label='SA-NOMA (Redundancy)')                                   # SA-NOMA redundância

    ax2.set_ylabel(r'Rate of Redundant Messages ($\rho$) [%]', fontsize=14)                        # Rótulo eixo Y direito

    # Título e legenda unificada
    plt.title("Impact of Number of Channels on Throughput and Redundancy", fontsize=14)             # Título do gráfico
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88), fontsize=10)                          # Legenda posicionada

    plt.tight_layout()                                                                              # Ajusta layout automaticamente

    # Salva gráfico no diretório de resultados (relativo ao script)
    script_dir = os.path.dirname(os.path.abspath(__file__))                                        # Diretório do script
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Channels_fig1.png')        # Caminho de saída
    plt.savefig(output_path, bbox_inches='tight', dpi=150)                                         # Salva em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")        # Log de confirmação

# Ponto de entrada: executa simulação quando o script é chamado diretamente
if __name__ == "__main__":
    run_simulation()                                 # Inicia a simulação principal