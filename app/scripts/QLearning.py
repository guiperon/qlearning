import numpy as np  # Importa NumPy para operações numéricas com arrays

def InitializeQTable(Devices, Relays, Slots, Runs, Initialization):
    """
    Inicializa a Q-Table (tabela de valores Q) usada pelo algoritmo Q-Learning.
    Cada entrada mapeia um par (canal, slot) para um valor de qualidade estimada.

    Args:
        Devices (int): Número total de dispositivos IoT na rede.
        Relays (int): Número de relays (canais de frequência disponíveis).
        Slots (int): Número de slots de tempo por quadro.
        Runs (int): Número de rodadas Monte Carlo independentes.
        Initialization (bool):
            True  -> Inicializa todos os valores Q com zero.
            False -> Inicializa com valores aleatórios uniformes no intervalo [-1, 1).

    Returns:
        numpy.ndarray: Matriz 4D com shape (Relays, Slots, Runs, Devices).
    """

    # Define as dimensões da Q-Table: (canais, slots, rodadas, dispositivos)
    shape = (Relays, Slots, Runs, Devices)

    if Initialization:
        c = np.zeros(shape)                    # Inicialização com zeros (exploração neutra)
    else:
        c = -1 + 2 * np.random.rand(*shape)   # Inicialização aleatória uniforme em [-1, 1)

    return c  # Retorna a Q-Table inicializada

def Qlearning_MultipleChannels(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula o protocolo Q-Learning com múltiplos canais e NOMA (SIC).
    Cada dispositivo aprende iterativamente a melhor combinação (canal, slot)
    usando a Q-Table, com recompensa +1 para sucesso e -1 para falha.

    Args:
        Devices (int): Número total de dispositivos IoT na rede.
        Relays (int): Número de relays (receptores) disponíveis.
        Channels_Relays (int): Número de canais de frequência disponíveis.
        runs (int): Número de rodadas Monte Carlo independentes.
        frames (int): Número de quadros (frames) de transmissão.
        Slots (int): Número de slots de tempo por quadro.
        SNR (numpy.ndarray): Matriz de SNR com shape (Devices, Relays, runs).
        N (float): Potência do ruído térmico (Watts).
        r (float): Taxa alvo mínima (bps/Hz) para decodificação com sucesso.
        QTable (numpy.ndarray): Tabela Q com shape (Relays, Slots, Runs, Devices).
        alpha (float): Taxa de aprendizado do Q-Learning (0 a 1).
        gamma (float): Fator de desconto temporal do Q-Learning (0 a 1).

    Returns:
        tuple: (ntput, ndist, ntotal) — throughput normalizado, tráfego distinto médio, tráfego total médio.
    """

    ThroughputRuns = []                                             # Lista para armazenar throughput médio de cada frame
    TotalTraffic = np.zeros(runs)                                   # Acumulador de tráfego total (com duplicatas) por rodada
    TotalTrafficDistinct = np.zeros(runs)                           # Acumulador de tráfego distinto (dispositivos únicos) por rodada

    # Pré-alocação das matrizes de escolha de ação (canal e slot) para cada dispositivo/rodada
    SlotChoosen = np.zeros((Devices, runs), dtype=int)              # Slot escolhido por cada dispositivo em cada rodada
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)           # Canal escolhido por cada dispositivo em cada rodada

    # --- Iteração sobre cada quadro de transmissão ---
    for l in range(frames):
        ThroughputFrame = np.zeros(runs)                            # Throughput acumulado neste frame por rodada
        SuccessTransmission = np.zeros((Devices, runs))             # Matriz de sucessos de transmissão por dispositivo/rodada

        # ==================================================================
        # 1. Busca Q-Learning: Escolha de Slot e Canal pela Q-Table
        # ==================================================================

        # Itera sobre cada dispositivo e rodada para selecionar a melhor ação
        for dd in range(Devices):
            for rr in range(runs):
                QTableDevice = QTable[:, :, rr, dd]                 # Extrai o plano Q (Canais x Slots) deste dispositivo/rodada

                maximum = np.max(QTableDevice)                      # Encontra o valor Q máximo no plano

                # Localiza todos os pares (canal, slot) que atingem o valor máximo
                y, x = np.where(QTableDevice == maximum)            # y = índices de canal, x = índices de slot

                if len(x) > 1:
                    # Empate: múltiplas ações com mesmo valor Q — escolhe aleatoriamente
                    randomChoice = np.random.randint(0, len(x))     # Índice aleatório entre as opções empatadas
                    ChannelChoosen[dd, rr] = y[randomChoice]        # Canal escolhido aleatoriamente entre os melhores
                    SlotChoosen[dd, rr] = x[randomChoice]           # Slot escolhido aleatoriamente entre os melhores
                else:
                    ChannelChoosen[dd, rr] = y[0]                   # Canal da única melhor ação
                    SlotChoosen[dd, rr] = x[0]                      # Slot da única melhor ação

        # Inicializa recompensas com -1 (penalidade padrão por falha na transmissão)
        Reward = -1 * np.ones((Devices, runs))

        # ==================================================================
        # 2. Processo de Transmissão (similar ao Slotted Aloha com NOMA)
        # ==================================================================

        # Itera sobre cada slot de tempo (índices 0 a Slots-1, compatível com Q-Table 0-indexed)
        for k in range(Slots):
            # --- Iteração Monte Carlo sobre cada rodada ---
            for s in range(runs):
                # Identifica quais dispositivos escolheram transmitir no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]       # Índices dos dispositivos ativos
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]     # Canal de cada dispositivo ativo

                # Verifica se há dispositivos transmitindo neste slot
                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]                  # SNR dos dispositivos ativos (Devices_ativos x Relays)
                    uniqueChannels = np.unique(TransmittingChannel)              # Lista de canais distintos em uso

                    # --- Processa cada canal separadamente ---
                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)                          # Máscara booleana: dispositivos neste canal
                        SNR_Device_Channel = SNR_Device[mask_channel, :]                   # SNR dos dispositivos neste canal
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]     # IDs dos dispositivos neste canal

                        # Ordena dispositivos do mais forte ao mais fraco para SIC-NOMA
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]                          # Índices de ordenação descendente por relay
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)        # SNR reordenado (mais forte primeiro)

                        # Mapeia IDs originais para a ordem SIC
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                        # --- Processa cada relay independentemente ---
                        for rr in range(Relays):
                            SIC_boolean = 0                                    # Flag de falha SIC: 0 = ativo, 1 = falhou
                            num_users_channel = SNR_Device_Channel.shape[0]    # Número de usuários colidindo neste canal

                            # --- Loop SIC: tenta decodificar do mais forte ao mais fraco ---
                            for jj in range(num_users_channel):
                                Interference = np.sum(SNR_Device_ord[jj+1:, rr])  # Interferência dos sinais mais fracos restantes
                                Signal = SNR_Device_ord[jj, rr]                    # Potência do sinal do usuário atual
                                SINR = Signal / (Interference + N)                 # Calcula SINR (sinal / interferência + ruído)

                                # Verifica se taxa alcançável supera o limiar e SIC ainda está ativo
                                if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                    ThroughputFrame[s] += 1                        # Incrementa throughput do frame

                                    device_id = TransmittingDevices_ord[jj, rr]    # ID original do dispositivo decodificado

                                    # Sucesso: define recompensa positiva e registra transmissão
                                    Reward[device_id, s] = 1                       # Recompensa +1 para atualização da Q-Table
                                    SuccessTransmission[device_id, s] += 1         # Contabiliza sucesso do dispositivo
                                else:
                                    SIC_boolean = 1                                # Marca falha no SIC — para decodificação subsequente

        # ==================================================================
        # 3. Atualização da Q-Table (Equação de Bellman modificada)
        # ==================================================================

        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]                    # Índice do canal escolhido por este dispositivo
                sl_idx = SlotChoosen[dd, rr]                       # Índice do slot escolhido por este dispositivo

                QTableDeviceVal = QTable[ch_idx, sl_idx, rr, dd]   # Valor Q atual da ação executada

                # Atualização Q: (1-α)*Q + α*(Recompensa + γ*Q) — fórmula fiel ao código MATLAB original
                ValorQTable = (1 - alpha) * QTableDeviceVal + alpha * (Reward[dd, rr] + gamma * QTableDeviceVal)

                QTable[ch_idx, sl_idx, rr, dd] = ValorQTable       # Atualiza a Q-Table com o novo valor

        # --- Estatísticas ao final de cada frame ---
        active_success = np.sum(SuccessTransmission > 0, axis=0)   # Dispositivos únicos com pelo menos 1 sucesso por rodada
        ThroughputRuns.append(np.mean(active_success) / Slots)     # Throughput médio normalizado por slots

        TotalTrafficDistinct += active_success                      # Acumula tráfego distinto ao longo dos frames
        TotalTraffic += np.sum(SuccessTransmission, axis=0)        # Acumula tráfego total (com duplicatas)

    # --- Cálculos finais de desempenho ---
    ndist = np.mean(TotalTrafficDistinct) / frames                 # Média de dispositivos distintos decodificados por frame
    ntotal = np.mean(TotalTraffic) / frames                        # Média de decodificações totais por frame

    # Evita divisão por zero caso não haja tráfego
    if ntotal == 0:
        ratio = 0                                                  # Razão nula se não houve tráfego
    else:
        ratio = ndist / ntotal                                     # Proporção de mensagens úteis (distintas / total)

    # Throughput normalizado: pondera pela taxa alvo, número de canais e razão de utilidade
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio

    return ntput, ndist, ntotal  # Retorna throughput normalizado, tráfego distinto e total

def Qlearning_MultipleChannels_NoNOMA(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula o protocolo Q-Learning com múltiplos canais SEM NOMA (apenas Efeito Captura).
    Diferente da versão NOMA, aqui apenas o dispositivo com sinal mais forte pode ser decodificado.

    Args:
        Devices (int): Número total de dispositivos IoT na rede.
        Relays (int): Número de relays (receptores) disponíveis.
        Channels_Relays (int): Número de canais de frequência disponíveis.
        runs (int): Número de rodadas Monte Carlo independentes.
        frames (int): Número de quadros de transmissão.
        Slots (int): Número de slots de tempo por quadro.
        SNR (numpy.ndarray): Matriz de SNR com shape (Devices, Relays, runs).
        N (float): Potência do ruído térmico (Watts).
        r (float): Taxa alvo mínima (bps/Hz) para decodificação com sucesso.
        QTable (numpy.ndarray): Tabela Q com shape (Relays, Slots, Runs, Devices).
        alpha (float): Taxa de aprendizado do Q-Learning (0 a 1).
        gamma (float): Fator de desconto temporal do Q-Learning (0 a 1).

    Returns:
        tuple: (ntput, ndist, ntotal) — throughput normalizado, tráfego distinto médio, tráfego total médio.
    """

    ThroughputRuns = []                                             # Lista para armazenar throughput médio de cada frame
    TotalTraffic = np.zeros(runs)                                   # Acumulador de tráfego total por rodada
    TotalTrafficDistinct = np.zeros(runs)                           # Acumulador de tráfego distinto por rodada

    # Pré-alocação das matrizes de escolha de ação
    SlotChoosen = np.zeros((Devices, runs), dtype=int)              # Slot escolhido por cada dispositivo/rodada
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)           # Canal escolhido por cada dispositivo/rodada

    # --- Iteração sobre cada quadro de transmissão ---
    for l in range(frames):
        ThroughputFrame = np.zeros(runs)                            # Throughput acumulado neste frame
        SuccessTransmission = np.zeros((Devices, runs))             # Matriz de sucessos por dispositivo/rodada

        # ==================================================================
        # 1. Busca Q-Learning: Escolha de Slot e Canal pela Q-Table
        # ==================================================================

        for dd in range(Devices):
            for rr in range(runs):
                QTableDevice = QTable[:, :, rr, dd]                 # Plano Q (Canais x Slots) deste dispositivo/rodada

                maximum = np.max(QTableDevice)                      # Valor Q máximo no plano

                # Localiza pares (canal, slot) que atingem o máximo
                y, x = np.where(QTableDevice == maximum)            # y = canais, x = slots

                if len(x) > 1:
                    randomChoice = np.random.randint(0, len(x))     # Índice aleatório para desempate
                    ChannelChoosen[dd, rr] = y[randomChoice]        # Canal escolhido aleatoriamente
                    SlotChoosen[dd, rr] = x[randomChoice]           # Slot escolhido aleatoriamente
                else:
                    ChannelChoosen[dd, rr] = y[0]                   # Canal da única melhor ação
                    SlotChoosen[dd, rr] = x[0]                      # Slot da única melhor ação

        # Inicializa recompensas com -1 (penalidade padrão por falha)
        Reward = -1 * np.ones((Devices, runs))

        # ==================================================================
        # 2. Processo de Transmissão (sem NOMA — apenas Efeito Captura)
        # ==================================================================

        # Itera sobre cada slot de tempo
        for k in range(Slots):
            for s in range(runs):
                # Identifica dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]       # Índices dos dispositivos ativos
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]     # Canal de cada dispositivo ativo

                # Verifica se há dispositivos transmitindo
                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]                  # SNR dos dispositivos ativos
                    uniqueChannels = np.unique(TransmittingChannel)              # Canais distintos em uso

                    # --- Processa cada canal separadamente ---
                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)                          # Máscara: dispositivos neste canal
                        SNR_Device_Channel = SNR_Device[mask_channel, :]                   # SNR neste canal
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]     # IDs neste canal

                        # Ordena do mais forte ao mais fraco para identificar o dominante
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]                          # Ordenação descendente
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)        # SNR reordenado
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]                   # IDs reordenados

                        # --- Processa cada relay ---
                        for rr in range(Relays):
                            SIC_boolean = 0                                    # Flag de falha (por compatibilidade estrutural)
                            num_users_channel = SNR_Device_Channel.shape[0]    # Número de usuários neste canal

                            # Sem NOMA: tenta decodificar APENAS o mais forte (range(1) = índice 0)
                            for jj in range(1):
                                if num_users_channel > 0:                      # Garante que há pelo menos 1 usuário
                                    Interference = np.sum(SNR_Device_ord[jj+1:, rr])  # Interferência dos demais
                                    Signal = SNR_Device_ord[jj, rr]                    # Sinal do mais forte
                                    SINR = Signal / (Interference + N)                 # Calcula SINR

                                    # Verifica se taxa alcançável supera o limiar
                                    if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                        ThroughputFrame[s] += 1                        # Incrementa throughput

                                        device_id = TransmittingDevices_ord[jj, rr]    # ID do dispositivo decodificado

                                        Reward[device_id, s] = 1                       # Recompensa +1 por sucesso
                                        SuccessTransmission[device_id, s] += 1         # Registra sucesso
                                    else:
                                        SIC_boolean = 1                                # Marca falha na decodificação

        # ==================================================================
        # 3. Atualização da Q-Table (Equação de Bellman modificada)
        # ==================================================================

        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]                    # Índice do canal escolhido
                sl_idx = SlotChoosen[dd, rr]                       # Índice do slot escolhido

                QTableDeviceVal = QTable[ch_idx, sl_idx, rr, dd]   # Valor Q atual da ação executada

                # Atualização Q: (1-α)*Q + α*(Recompensa + γ*Q) — fiel ao MATLAB original
                ValorQTable = (1 - alpha) * QTableDeviceVal + alpha * (Reward[dd, rr] + gamma * QTableDeviceVal)

                QTable[ch_idx, sl_idx, rr, dd] = ValorQTable       # Atualiza Q-Table

        # --- Estatísticas ao final de cada frame ---
        active_success = np.sum(SuccessTransmission > 0, axis=0)   # Dispositivos únicos com sucesso por rodada
        ThroughputRuns.append(np.mean(active_success) / Slots)     # Throughput normalizado médio

        TotalTrafficDistinct += active_success                      # Acumula tráfego distinto
        TotalTraffic += np.sum(SuccessTransmission, axis=0)        # Acumula tráfego total

    # --- Cálculos finais de desempenho ---
    ndist = np.mean(TotalTrafficDistinct) / frames                 # Média de dispositivos distintos por frame
    ntotal = np.mean(TotalTraffic) / frames                        # Média de decodificações totais por frame

    # Evita divisão por zero
    if ntotal == 0:
        ratio = 0                                                  # Sem tráfego, razão é zero
    else:
        ratio = ndist / ntotal                                     # Proporção de mensagens úteis

    # Throughput normalizado final
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio

    return ntput, ndist, ntotal  # Retorna throughput normalizado, tráfego distinto e total


def Qlearning_UniqueChannel(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula Q-Learning com Canal Único (Canais/Relays Ortogonais).
    Diferente da versão MultipleChannels, aqui o dispositivo escolhe um canal específico
    e seu sinal é recebido exclusivamente pelo relay correspondente àquele canal.

    Args:
        Devices (int): Número total de dispositivos IoT na rede.
        Relays (int): Número de relays disponíveis.
        Channels_Relays (int): Número de canais de frequência disponíveis.
        runs (int): Número de rodadas Monte Carlo independentes.
        frames (int): Número de quadros de transmissão.
        Slots (int): Número de slots de tempo por quadro.
        SNR (numpy.ndarray): Matriz de SNR com shape (Devices, Relays, runs).
        N (float): Potência do ruído térmico (Watts).
        r (float): Taxa alvo mínima (bps/Hz) para decodificação com sucesso.
        QTable (numpy.ndarray): Tabela Q com shape (Channels, Slots, Runs, Devices).
        alpha (float): Taxa de aprendizado do Q-Learning (0 a 1).
        gamma (float): Fator de desconto temporal do Q-Learning (0 a 1).

    Returns:
        tuple: (ntput, ndist, ntotal) — throughput normalizado, tráfego distinto médio, tráfego total médio.
    """

    ThroughputRuns = []                                             # Lista para armazenar throughput médio de cada frame
    TotalTraffic = np.zeros(runs)                                   # Acumulador de tráfego total por rodada
    TotalTrafficDistinct = np.zeros(runs)                           # Acumulador de tráfego distinto por rodada

    # Pré-alocação das matrizes de escolha de ação
    SlotChoosen = np.zeros((Devices, runs), dtype=int)              # Slot escolhido por cada dispositivo/rodada
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)           # Canal escolhido por cada dispositivo/rodada

    # --- Iteração sobre cada quadro de transmissão ---
    for l in range(frames):
        ThroughputFrame = np.zeros(runs)                            # Throughput acumulado neste frame
        SuccessTransmission = np.zeros((Devices, runs))             # Matriz de sucessos por dispositivo/rodada

        # ==================================================================
        # 1. Busca Q-Learning: Escolha de Ação pela Q-Table
        # ==================================================================
        for dd in range(Devices):
            for rr in range(runs):
                QTableDevice = QTable[:, :, rr, dd]                 # Plano Q (Canais x Slots) deste dispositivo/rodada

                maximum = np.max(QTableDevice)                      # Valor Q máximo no plano

                # Localiza pares (canal, slot) que atingem o máximo
                y, x = np.where(QTableDevice == maximum)            # y = canais, x = slots

                if len(x) > 1:
                    rand_idx = np.random.randint(0, len(x))         # Índice aleatório para desempate
                    ChannelChoosen[dd, rr] = y[rand_idx]            # Canal escolhido aleatoriamente
                    SlotChoosen[dd, rr] = x[rand_idx]               # Slot escolhido aleatoriamente
                else:
                    ChannelChoosen[dd, rr] = y[0]                   # Canal da única melhor ação
                    SlotChoosen[dd, rr] = x[0]                      # Slot da única melhor ação

        # Inicializa recompensas com -1 (penalidade padrão por falha)
        Reward = -1 * np.ones((Devices, runs))

        # ==================================================================
        # 2. Processo de Transmissão (Canal Único / Ortogonal)
        # ==================================================================
        for k in range(Slots):
            for s in range(runs):
                # Identifica dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]       # Índices dos dispositivos ativos
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]     # Canal de cada dispositivo ativo

                # Verifica se há dispositivos transmitindo
                if len(TransmittingDevices) >= 1:
                    # --- Extração de SNR específico do canal escolhido (lógica Unique Channel) ---
                    # Cada dispositivo usa apenas o SNR do relay correspondente ao seu canal
                    SNR_Device = SNR[TransmittingDevices, TransmittingChannel, s]    # SNR 1D: valor por dispositivo

                    uniqueChannels = np.unique(TransmittingChannel)                  # Canais distintos em uso

                    # --- Processa cada canal separadamente ---
                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)                    # Máscara: dispositivos neste canal

                        SNR_Device_Channel = SNR_Device[mask_channel]                # SNR dos dispositivos neste canal (1D)
                        TransmittingDevices_Channel = TransmittingDevices[mask_channel]  # IDs dos dispositivos neste canal

                        # Ordena do mais forte ao mais fraco para SIC
                        sort_indexes = np.argsort(SNR_Device_Channel)[::-1]          # Ordenação descendente (1D)

                        SNR_Device_ord = SNR_Device_Channel[sort_indexes]            # SNR reordenado
                        TransmittingDevices_ord = TransmittingDevices_Channel[sort_indexes]  # IDs reordenados

                        # --- Loop SIC: tenta decodificar do mais forte ao mais fraco ---
                        SIC_boolean = 0                                              # Flag de falha SIC
                        num_users = len(SNR_Device_ord)                              # Número de usuários neste canal

                        for jj in range(num_users):
                            Interference = np.sum(SNR_Device_ord[jj+1:])             # Interferência dos sinais mais fracos
                            Signal = SNR_Device_ord[jj]                              # Sinal do usuário atual
                            SINR = Signal / (Interference + N)                       # Calcula SINR

                            # Verifica se taxa alcançável supera o limiar e SIC está ativo
                            if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                ThroughputFrame[s] += 1                              # Incrementa throughput

                                dev_id = TransmittingDevices_ord[jj]                 # ID do dispositivo decodificado
                                Reward[dev_id, s] = 1                                # Recompensa +1 por sucesso
                                SuccessTransmission[dev_id, s] += 1                  # Registra sucesso
                            else:
                                SIC_boolean = 1                                      # Falha no SIC — para decodificação

        # ==================================================================
        # 3. Atualização da Q-Table (Equação de Bellman modificada)
        # ==================================================================
        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]                    # Índice do canal escolhido
                sl_idx = SlotChoosen[dd, rr]                       # Índice do slot escolhido

                current_q = QTable[ch_idx, sl_idx, rr, dd]        # Valor Q atual da ação executada

                # Atualização Q: (1-α)*Q + α*(Recompensa + γ*Q) — fiel ao MATLAB original
                new_q = (1 - alpha) * current_q + alpha * (Reward[dd, rr] + gamma * current_q)

                QTable[ch_idx, sl_idx, rr, dd] = new_q            # Atualiza Q-Table com novo valor

        # --- Estatísticas ao final de cada frame ---
        active_success = np.sum(SuccessTransmission > 0, axis=0)   # Dispositivos únicos com sucesso por rodada
        ThroughputRuns.append(np.mean(active_success) / Slots)     # Throughput normalizado médio

        TotalTrafficDistinct += active_success                      # Acumula tráfego distinto
        TotalTraffic += np.sum(SuccessTransmission, axis=0)        # Acumula tráfego total

    # --- Cálculos finais de desempenho ---
    ndist = np.mean(TotalTrafficDistinct) / frames                 # Média de dispositivos distintos por frame
    ntotal = np.mean(TotalTraffic) / frames                        # Média de decodificações totais por frame

    # Evita divisão por zero
    if ntotal == 0:
        ratio = 0                                                  # Sem tráfego, razão é zero
    else:
        ratio = ndist / ntotal                                     # Proporção de mensagens úteis

    # Throughput normalizado final
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio

    return ntput, ndist, ntotal  # Retorna throughput normalizado, tráfego distinto e total