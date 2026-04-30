import numpy as np                                   # Importa NumPy para operações numéricas com arrays
import matplotlib.pyplot as plt                      # Importa interface de plotagem do Matplotlib
from datetime import datetime                        # Importa datetime para timestamps nas mensagens de log
import os                                            # Importa os para manipulação de caminhos e processos
from concurrent.futures import ProcessPoolExecutor, as_completed  # Importa executor de processos paralelos

from StochasticGeometry import StochasticGeometry    # Importa gerador de geometria estocástica
from SlottedAloha import SlottedAloha_MultipleChannels  # Importa simulador Slotted Aloha com NOMA
from QLearning import InitializeQTable, Qlearning_MultipleChannels, Qlearning_UniqueChannel  # Importa funções Q-Learning

# ==============================================================================
# WORKER — executa uma iteração de relays em seu próprio processo
# ==============================================================================
def _run_relays_iteration(args):
    """
    Função worker que simula uma única configuração de número de relays/canais.
    Executada em processo separado pelo ProcessPoolExecutor.
    Nesta simulação, o número de relays é igual ao número de canais (R = C).

    Args:
        args (tuple): Tupla com todos os parâmetros necessários.

    Returns:
        tuple: (índice, resultados SA, QL-NOMA e QL-UniqueChannel).
    """
    # Desempacota parâmetros recebidos
    (idx, n_relays, Channels_Relays, Devices, runs, frames, slots,
     N, r, alpha, gamma, P, Distance, h_Nakagami) = args

    # Exibe mensagem de progresso com timestamp e PID
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"[PID {os.getpid()}] Relays: {n_relays}")

    # Fatia as matrizes de distância e canal para o número atual de relays
    dist_slice = Distance[:, :n_relays, :]           # Distâncias apenas para os n_relays primeiros relays
    h_slice = h_Nakagami[:, :n_relays, :]            # Canal Nakagami apenas para os n_relays primeiros relays

    # Calcula atenuação de percurso (path loss) usando modelo COST-231
    alpha_k_j = 10**(-(128.1 + 36.7 * np.log10(dist_slice)) / 10)
    # Calcula SNR recebida para a configuração atual
    SNR = (P / N * alpha_k_j * h_slice**2)

    # --- Slotted Aloha com NOMA ---
    tp_sa, nd_sa, nt_sa = SlottedAloha_MultipleChannels(
        Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r
    )

    # --- Q-Learning com NOMA (múltiplos canais) ---
    QTable = InitializeQTable(Devices, Channels_Relays, slots, runs, True)   # Q-Table inicializada com zeros
    tp_ql, nd_ql, nt_ql = Qlearning_MultipleChannels(
        Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
    )

    # --- Q-Learning com Canal Único (ortogonal) ---
    QTable = InitializeQTable(Devices, Channels_Relays, slots, runs, True)   # Reinicializa Q-Table
    tp_ql_unique, nd_ql_unique, nt_ql_unique = Qlearning_UniqueChannel(
        Devices, n_relays, Channels_Relays, runs, frames, slots, SNR, N, r, QTable, alpha, gamma
    )

    # Retorna índice e todos os resultados dos 3 métodos
    return idx, tp_sa, nd_sa, nt_sa, tp_ql, nd_ql, nt_ql, tp_ql_unique, nd_ql_unique, nt_ql_unique

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_simulation():
    """
    Função principal que configura parâmetros, executa simulações em paralelo
    e gera gráficos de redundância e throughput vs número de relays/canais.
    """
    # Configurações dos parâmetros
    np.random.seed(0)                                # Fixa semente aleatória para reprodutibilidade

    # --- Parâmetros dos Dispositivos ---
    Devices = 1500                                   # Número fixo de dispositivos IoT

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

    # --- Faixa de Relays/Canais (variável principal: R = C de 1 a 15) ---
    Relays_Range = np.arange(1, 16)                  # Número de relays: 1 a 15
    Channels_Relays_Range = Relays_Range             # Número de canais = número de relays (R = C)

    r = 3                                            # Taxa alvo mínima (bps/Hz)

    # --- Parâmetros de Geometria Estocástica ---
    cell_radius = 5e3                                # Raio da célula (5 km)

    # --- Parâmetros de Satélite ---
    altitude = 780e3                                 # Altitude do satélite (780 km)

    # --- Parâmetros de Simulação ---
    runs = 100                                       # Rodadas Monte Carlo
    slots = 100                                      # Slots de tempo por quadro
    frames = 10                                      # Quadros de transmissão

    alpha = 0.1                                      # Taxa de aprendizado Q-Learning
    gamma = 0.5                                      # Fator de desconto Q-Learning

    # --- Canal Nakagami-m ---
    m = 2                                            # Parâmetro Nakagami (desvanecimento moderado)

    # --- Geometria Estocástica: gera posições para o máximo de relays ---
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gerando Geometria Estocástica...")

    max_relays = np.max(Relays_Range)                # Máximo de relays para pré-alocar matrizes

    Distance = StochasticGeometry(Devices, max_relays, cell_radius, runs)  # Distâncias (Devices x max_relays x runs)

    # Geração do canal Nakagami complexo
    shape_k = m / 2                                  # Parâmetro shape da distribuição Gamma
    scale_theta = 1                                  # Parâmetro scale da distribuição Gamma
    size_h = (Devices, max_relays, runs)             # Dimensões para o máximo de relays

    h_real = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte real do canal
    h_imag = np.sqrt(np.random.gamma(shape_k, scale_theta, size_h))    # Parte imaginária do canal
    h_Nakagami = np.abs((h_real + 1j * h_imag) / np.sqrt(m))           # Módulo normalizado

    n_cases = len(Relays_Range)                      # Número de configurações a simular

    # --- Pré-alocação das estruturas de resultados ---
    res_sa = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}           # Resultados SA-NOMA
    res_ql = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}           # Resultados QL-NOMA
    res_ql_unique = {'ntput': np.zeros(n_cases), 'ndist': np.zeros(n_cases), 'ntotal': np.zeros(n_cases)}    # Resultados QL-UC

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Loop de Simulação...")

    # Monta argumentos para cada configuração de relays
    task_args = [
        (idx, int(n_relays), int(Channels_Relays_Range[idx]), Devices, runs, frames, slots,
         N, r, alpha, gamma, P, Distance, h_Nakagami)
        for idx, n_relays in enumerate(Relays_Range)
    ]

    # Determina número de workers paralelos
    n_workers = min(os.cpu_count() or 1, n_cases)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Using {n_workers} parallel workers for {n_cases} relays cases.")

    # --- Execução paralela: distribui cada configuração para um processo ---
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_relays_iteration, args): args[0] for args in task_args}  # Submete tarefas
        for future in as_completed(futures):                                                      # Coleta resultados
            (idx, tp_sa, nd_sa, nt_sa,
             tp_ql, nd_ql, nt_ql,
             tp_ql_unique, nd_ql_unique, nt_ql_unique) = future.result()                          # Desempacota
            res_sa['ntput'][idx] = tp_sa                                                          # Throughput SA-NOMA
            res_sa['ndist'][idx] = nd_sa                                                          # Tráfego distinto SA
            res_sa['ntotal'][idx] = nt_sa                                                         # Tráfego total SA
            res_ql['ntput'][idx] = tp_ql                                                          # Throughput QL-NOMA
            res_ql['ndist'][idx] = nd_ql                                                          # Tráfego distinto QL
            res_ql['ntotal'][idx] = nt_ql                                                         # Tráfego total QL
            res_ql_unique['ntput'][idx] = tp_ql_unique                                            # Throughput QL-UC
            res_ql_unique['ndist'][idx] = nd_ql_unique                                            # Tráfego distinto QL-UC
            res_ql_unique['ntotal'][idx] = nt_ql_unique                                           # Tráfego total QL-UC

    # --- Função auxiliar para calcular taxa de redundância ---
    def calc_red(r_dict):
        """Calcula taxa de redundância: 100*(1 - distinto/total), tratando NaN."""
        nd = np.array(r_dict['ndist'])               # Array de tráfego distinto
        nt = np.array(r_dict['ntotal'])              # Array de tráfego total
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignora warnings de divisão por zero
            red = 100 * (1 - nd / nt)               # Calcula redundância percentual
        return np.nan_to_num(red)                    # Substitui NaN por 0

    red_sa = calc_red(res_sa)                        # Taxa de redundância SA-NOMA
    red_ql = calc_red(res_ql)                        # Taxa de redundância QL-NOMA
    red_ql_unique = calc_red(res_ql_unique)          # Taxa de redundância QL-UniqueChannel

    # --- Figura 1: Gráfico de Barras — Taxa de Redundância ---
    plt.figure(1, figsize=(10, 6))                                   # Cria figura 10x6

    width = 0.25                                                     # Largura de cada barra no gráfico agrupado
    x = np.arange(len(Relays_Range))                                 # Posições base no eixo X

    plt.bar(x - width, red_sa, width, label='SA-NOMA', color='#0072BD')           # Barras SA-NOMA (azul)
    plt.bar(x, red_ql, width, label='QL-NOMA', color='#D95319')                   # Barras QL-NOMA (laranja)
    plt.bar(x + width, red_ql_unique, width, label='QL Orthogonal', color='#EDB120')  # Barras QL-UC (amarelo)

    plt.xlabel('Number of Relays = Number of Channels', fontsize=13)              # Rótulo eixo X
    plt.ylabel('Rate of Redundant Messages (%)', fontsize=13)                     # Rótulo eixo Y
    plt.legend()                                                                   # Exibe legenda
    plt.grid(True, axis='y')                                                       # Grade apenas no eixo Y
    plt.xticks(x, Relays_Range)                                                   # Labels corretos no eixo X
    plt.xlim([-0.5, len(Relays_Range) - 0.5])                                    # Limites do eixo X
    plt.title('Redundancy Rate Comparison')                                        # Título do gráfico
    plt.tight_layout()                                                             # Ajusta layout

    # --- Figura 2: Gráfico de Linhas — Throughput Normalizado ---
    plt.figure(2, figsize=(10, 6))                                                # Cria segunda figura

    plt.plot(Relays_Range, res_ql['ntput'], 'o-', linewidth=1.5, markersize=10,
             markerfacecolor='w', label='QL-NOMA')                                 # Curva QL-NOMA
    plt.plot(Relays_Range, res_ql_unique['ntput'], 's-', linewidth=1.5, markersize=10,
             markerfacecolor='w', label='DQL-based JRSAC [4]')                     # Curva QL-UniqueChannel
    plt.plot(Relays_Range, res_sa['ntput'], '*-', linewidth=1.5, markersize=10,
             markerfacecolor='w', label='SA-NOMA')                                 # Curva SA-NOMA

    plt.grid(True)                                                                 # Ativa grade
    plt.xlabel('Number of Relays = Number of Channels (R = C)', fontsize=14)      # Rótulo eixo X
    plt.ylabel(r'Normalized Throughput ($\tau$) [bps/Hz]', fontsize=14)           # Rótulo eixo Y com LaTeX
    plt.legend(fontsize=12)                                                        # Legenda
    plt.title('Normalized Throughput Comparison')                                  # Título
    plt.tight_layout()                                                             # Ajusta layout

    # Salva gráfico no diretório de resultados (relativo ao script)
    script_dir = os.path.dirname(os.path.abspath(__file__))                       # Diretório do script
    output_path = os.path.join(script_dir, '..', 'results', 'Throughput_Channels_Relays_fig1.png')  # Caminho de saída
    plt.savefig(output_path, bbox_inches='tight', dpi=150)                        # Salva em PNG (150 DPI)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved plot to: {output_path}")  # Log de confirmação

# Ponto de entrada: executa simulação quando o script é chamado diretamente
if __name__ == "__main__":
    run_simulation()                                 # Inicia a simulação principal