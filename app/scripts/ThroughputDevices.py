import numpy as np                                   # Importa NumPy para operações numéricas com arrays
import matplotlib.pyplot as plt                      # Importa interface de plotagem do Matplotlib
from datetime import datetime                        # Importa datetime para timestamps nas mensagens de log
import os                                            # Importa os para manipulação de caminhos e processos
from concurrent.futures import ProcessPoolExecutor, as_completed  # Importa executor de processos paralelos

from StochasticGeometry import StochasticGeometry    # Importa gerador de geometria estocástica
from SlottedAloha import SlottedAloha_MultipleChannels, SlottedAloha_MultipleChannels_NoNOMA  # Importa simuladores SA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_MultipleChannels_NoNOMA  # Importa funções QL

# ==============================================================================
# WORKER — executa uma iteração de dispositivos em seu próprio processo
# ==============================================================================
def _run_devices_iteration(args):
    """
    Função worker que simula uma única configuração de número de dispositivos.
    Para cada quantidade de dispositivos, executa SA e QL (com e sem NOMA) para múltiplos canais.
    Executada em processo separado pelo ProcessPoolExecutor.

    Args:
        args (tuple): Tupla com todos os parâmetros necessários.

    Returns:
        tuple: (índice, dicionário de resultados por canal).
    """
    # Desempacota parâmetros recebidos
    (idx, num_dev, Channels_Relays_List, Relays, runs, frames, slots,
     N, r, alpha, gamma, P, Distance, h_Nakagami) = args

    # Exibe mensagem de progresso com timestamp e PID
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Devices: {num_dev}")

    # Fatia as matrizes para o número atual de dispositivos
    dist_slice = Distance[:num_dev, :, :]            # Distâncias apenas para os num_dev primeiros dispositivos
    h_slice = h_Nakagami[:num_dev, :, :]             # Canal Nakagami para os num_dev primeiros dispositivos

    # Calcula atenuação de percurso (path loss) usando modelo COST-231
    alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(dist_slice)) / 10)
    # Calcula SNR recebida
    SNR = (P / N * alpha_k_j * h_slice**2)

    per_channel_results = {}                         # Dicionário para armazenar resultados por número de canais

    # Itera sobre cada configuração de número de canais
    for ch_count in Channels_Relays_List:
        # --- Slotted Aloha com NOMA ---
        tp, dist, total = SlottedAloha_MultipleChannels(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r
        )

        # --- Slotted Aloha sem NOMA (apenas Efeito Captura) ---
        tp_nonoma, dist_nonoma, total_nonoma = SlottedAloha_MultipleChannels_NoNOMA(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r
        )

        # --- Q-Learning com NOMA ---
        QTable = InitializeQTable(num_dev, ch_count, slots, runs, True)     # Q-Table inicializada com zeros
        tp_ql, dist_ql, total_ql = Qlearning_MultipleChannels(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
        )

        # --- Q-Learning sem NOMA ---
        QTable = InitializeQTable(num_dev, ch_count, slots, runs, True)     # Reinicializa Q-Table
        tp_ql_nonoma, dist_ql_nonoma, total_ql_nonoma = Qlearning_MultipleChannels_NoNOMA(
            num_dev, Relays, ch_count, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
        )

        # Armazena resultados dos 4 métodos para este número de canais
        per_channel_results[ch_count] = {
            'SA': (tp, dist, total),                                        # Resultados SA-NOMA
            'SA_NoNOMA': (tp_nonoma, dist_nonoma, total_nonoma),            # Resultados SA sem NOMA
            'QL': (tp_ql, dist_ql, total_ql),                              # Resultados QL-NOMA
            'QL_NoNOMA': (tp_ql_nonoma, dist_ql_nonoma, total_ql_nonoma),  # Resultados QL sem NOMA
        }

    return idx, per_channel_results  # Retorna índice e resultados por canal

# ==============================================================================
# SCRIPT PRINCIPAL (Main)
# ==============================================================================

def run_simulation():
    """
    Função principal que configura parâmetros, executa simulações em paralelo
    e gera gráficos de throughput e redundância vs número de dispositivos.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Simulação...")

    # Configurações dos parâmetros
    np.random.seed(0)                                # Fixa semente aleatória para reprodutibilidade

    # --- Parâmetros dos Dispositivos (variável principal) ---
    Devices = np.linspace(100, 1500, 15, dtype=int)  # 15 valores de 100 a 1500 dispositivos

    # --- Potência de Transmissão ---
    P_dBm = 10                                       # Potência fixa em dBm
    P = 10**((P_dBm - 30) / 10)                     # Converte para Watts

    # --- Parâmetros de Ruído ---
    FdB = 6                                          # Figura de ruído (dB)
    F = 10**(FdB / 10)                               # Figura de ruído (linear)
    N0dB = -204                                      # Densidade espectral de ruído (dBW/Hz)
    N0 = 10**(N0dB / 10)                             # Densidade de ruído (Watts/Hz)
    B = 100e3                                        # Largura de banda (100 kHz)
    N = N0 * B * F                                   # Potência total do ruído

    # --- Parâmetros de Relays e Canais ---
    Relays = 4                                       # Número fixo de relays
    Channels_Relays_List = [1, 2, 3, 4]              # Configurações de canais a simular

    r = 3                                            # Taxa alvo mínima (bps/Hz)

    # --- Parâmetros de Geometria Estocástica ---
    cell_radius = 5e3                                # Raio da célula (5 km)

    # --- Parâmetros de Simulação ---
    runs = 100                                       # Rodadas Monte Carlo
    slots = 100                                      # Slots de tempo por quadro
    frames = 50                                      # Quadros de transmissão

    alpha = 0.1                                      # Taxa de aprendizado Q-Learning
    gamma = 0.5                                      # Fator de desconto Q-Learning

    # --- Canal Nakagami-m ---
    m = 2                                            # Parâmetro Nakagami (desvanecimento moderado)

    # --- Geometria Estocástica: gera posições para o máximo de dispositivos ---
    max_devs = np.max(Devices)                       # Máximo de dispositivos (para pré-alocar)
    max_relays = Relays                              # Número fixo de relays (4)

    Distance = StochasticGeometry(max_devs, max_relays, cell_radius, runs)  # Distâncias (max_devs x Relays x runs)

    # Geração do canal Nakagami complexo
    shape_k = m / 2                                  # Parâmetro shape da distribuição Gamma
    scale_theta = 1                                  # Parâmetro scale da distribuição Gamma
    size_h = (max_devs, max_relays, runs)            # Dimensões para o máximo de dispositivos

    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte real do canal
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte imaginária do canal
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))           # Módulo normalizado

    # --- Pré-alocação das estruturas de resultados ---
    # Dicionário aninhado: res_data[método][num_canais] = {ntput, ndist, ntotal}
    methods = ['SA', 'SA_NoNOMA', 'QL', 'QL_NoNOMA']                   # Lista de métodos simulados
    n_devices = len(Devices)                                             # Número de cenários de dispositivos
    res_data = {
        method: {
            ch: {'ntput': np.zeros(n_devices), 'ndist': np.zeros(n_devices), 'ntotal': np.zeros(n_devices)}
            for ch in Channels_Relays_List                               # Para cada configuração de canais
        }
        for method in methods                                            # Para cada método (SA, QL, com/sem NOMA)
    }

    # Monta argumentos para cada quantidade de dispositivos
    task_args = [
        (idx, int(num_dev), Channels_Relays_List, Relays, runs, frames, slots,
         N, r, alpha, gamma, P, Distance, h_Nakagami)
        for idx, num_dev in enumerate(Devices)
    ]

    # Determina número de workers paralelos
    n_workers = min(os.cpu_count() or 1, n_devices)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_devices} device cases.")

    # --- Execução paralela: distribui cada cenário de dispositivos para um processo ---
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_devices_iteration, args): args[0] for args in task_args}  # Submete tarefas
        for future in as_completed(futures):                                                       # Coleta resultados
            idx, per_channel_results = future.result()                                             # Desempacota resultado
            for ch_count in Channels_Relays_List:                                                  # Para cada configuração de canais
                # Desempacota resultados SA-NOMA
                sa_tp, sa_dist, sa_total = per_channel_results[ch_count]['SA']
                # Desempacota resultados SA sem NOMA
                sa_no_tp, sa_no_dist, sa_no_total = per_channel_results[ch_count]['SA_NoNOMA']
                # Desempacota resultados QL-NOMA
                ql_tp, ql_dist, ql_total = per_channel_results[ch_count]['QL']
                # Desempacota resultados QL sem NOMA
                ql_no_tp, ql_no_dist, ql_no_total = per_channel_results[ch_count]['QL_NoNOMA']

                # Armazena resultados SA-NOMA na estrutura de dados
                res_data['SA'][ch_count]['ntput'][idx] = sa_tp                # Throughput SA-NOMA
                res_data['SA'][ch_count]['ndist'][idx] = sa_dist              # Tráfego distinto SA
                res_data['SA'][ch_count]['ntotal'][idx] = sa_total            # Tráfego total SA

                # Armazena resultados SA sem NOMA
                res_data['SA_NoNOMA'][ch_count]['ntput'][idx] = sa_no_tp      # Throughput SA sem NOMA
                res_data['SA_NoNOMA'][ch_count]['ndist'][idx] = sa_no_dist    # Tráfego distinto SA sem NOMA
                res_data['SA_NoNOMA'][ch_count]['ntotal'][idx] = sa_no_total  # Tráfego total SA sem NOMA

                # Armazena resultados QL-NOMA
                res_data['QL'][ch_count]['ntput'][idx] = ql_tp                # Throughput QL-NOMA
                res_data['QL'][ch_count]['ndist'][idx] = ql_dist              # Tráfego distinto QL
                res_data['QL'][ch_count]['ntotal'][idx] = ql_total            # Tráfego total QL

                # Armazena resultados QL sem NOMA
                res_data['QL_NoNOMA'][ch_count]['ntput'][idx] = ql_no_tp      # Throughput QL sem NOMA
                res_data['QL_NoNOMA'][ch_count]['ndist'][idx] = ql_no_dist    # Tráfego distinto QL sem NOMA
                res_data['QL_NoNOMA'][ch_count]['ntotal'][idx] = ql_no_total  # Tráfego total QL sem NOMA

    # --- Cálculo das taxas de redundância: 100*(1 - distinto/total) ---
    redundancy = {m: {ch: [] for ch in Channels_Relays_List} for m in methods}  # Estrutura para redundância

    for m in methods:                                                    # Para cada método
        for ch in Channels_Relays_List:                                  # Para cada configuração de canais
            ndist_arr = np.array(res_data[m][ch]['ndist'])               # Array de tráfego distinto
            ntotal_arr = np.array(res_data[m][ch]['ntotal'])             # Array de tráfego total

            with np.errstate(divide='ignore', invalid='ignore'):         # Ignora warnings de divisão por zero
                red = 100 * (1 - ndist_arr / ntotal_arr)                 # Taxa de redundância percentual
                red = np.nan_to_num(red)                                 # Substitui NaN por 0

            redundancy[m][ch] = red                                      # Armazena redundância calculada

    # --- Seleção do canal para plotagem (canal 1 conforme código original) ---
    ch_to_plot = 1                                                       # Número de canais para o gráfico principal

    devices_plot_idx = range(len(Devices))                               # Índices de todos os pontos a plotar

    # --- FIGURA 1: Throughput Normalizado vs Número de Dispositivos ---
    plt.figure(1, figsize=(8, 6))                                        # Cria figura 8x6

    plt.plot(Devices, res_data['QL'][ch_to_plot]['ntput'], 'o-', linewidth=1.5, markersize=8,
             markerfacecolor='w', label='QL-NOMA')                        # Curva QL-NOMA
    plt.plot(Devices, res_data['SA'][ch_to_plot]['ntput'], '*-', linewidth=1.5, markersize=8,
             label='SA-NOMA')                                             # Curva SA-NOMA

    plt.plot(Devices, res_data['QL_NoNOMA'][ch_to_plot]['ntput'], 'o:', linewidth=1.5, markersize=8,
             markerfacecolor='w', label='QL without NOMA')                # Curva QL sem NOMA (pontilhado)
    plt.plot(Devices, res_data['SA_NoNOMA'][ch_to_plot]['ntput'], '*:', linewidth=1.5, markersize=8,
             label='SA without NOMA')                                     # Curva SA sem NOMA (pontilhado)

    plt.grid(True)                                                        # Ativa grade
    plt.xlabel('Number of Devices (D)', fontsize=14)                     # Rótulo eixo X
    plt.ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)  # Rótulo eixo Y com LaTeX
    plt.legend(fontsize=12)                                               # Legenda
    plt.title(f'Throughput ({ch_to_plot} Channel)', fontsize=14)         # Título com número de canais
    plt.tight_layout()                                                    # Ajusta layout

    # Salva figura 1 no diretório de resultados
    script_dir = os.path.dirname(os.path.abspath(__file__))              # Diretório do script
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Devices_fig1.png')  # Caminho de saída
    plt.savefig(output_path, bbox_inches='tight', dpi=150)               # Salva em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")  # Log

    # --- FIGURA 2: Taxa de Redundância vs Número de Dispositivos ---
    plt.figure(2, figsize=(8, 6))                                        # Cria segunda figura

    plt.plot(Devices, redundancy['QL'][ch_to_plot], 'o-', linewidth=1.5, markersize=8,
             markerfacecolor='w', label='QL-NOMA')                        # Curva QL-NOMA
    plt.plot(Devices, redundancy['SA'][ch_to_plot], '*-', linewidth=1.5, markersize=8,
             label='SA-NOMA')                                             # Curva SA-NOMA
    plt.plot(Devices, redundancy['QL_NoNOMA'][ch_to_plot], 'o:', linewidth=1.5, markersize=8,
             markerfacecolor='w', label='QL without NOMA')                # Curva QL sem NOMA
    plt.plot(Devices, redundancy['SA_NoNOMA'][ch_to_plot], '*:', linewidth=1.5, markersize=8,
             label='SA without NOMA')                                     # Curva SA sem NOMA

    plt.grid(True)                                                        # Ativa grade
    plt.xlabel('Number of Devices (D)', fontsize=14)                     # Rótulo eixo X
    plt.ylabel(r'Rate of Redundant Messages ($\rho$) [%]', fontsize=14) # Rótulo eixo Y com LaTeX
    plt.legend(fontsize=12)                                               # Legenda
    plt.title(f'Redundancy ({ch_to_plot} Channel)', fontsize=14)         # Título
    plt.tight_layout()                                                    # Ajusta layout

    # Salva figura 2 no diretório de resultados
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Devices_fig2.png')  # Caminho de saída
    plt.savefig(output_path, bbox_inches='tight', dpi=150)               # Salva em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")  # Log

# Ponto de entrada: executa simulação quando o script é chamado diretamente
if __name__ == "__main__":
    run_simulation()                                 # Inicia a simulação principal