import numpy as np

def InitializeQTable(Devices, Relays, Slots, Runs, Initialization):
    """
    Inicializa a Q-Table com zeros ou valores aleatórios entre -1 e 1.
    
    Args:
        Devices (int): Número de dispositivos.
        Relays (int): Número de relés (canais).
        Slots (int): Número de slots de tempo.
        Runs (int): Número de rodadas de simulação.
        Initialization (bool): 
            True  -> Inicializa com Zeros.
            False -> Inicializa com valores aleatórios uniformes [-1, 1].
            
    Returns:
        numpy.ndarray: Matriz 4D (Relays x Slots x Runs x Devices).
    """
    
    # Definir as dimensões da matriz
    # Nota: A ordem das dimensões foi mantida igual ao MATLAB:
    # (Relays, Slots, Runs, Devices)
    shape = (Relays, Slots, Runs, Devices)

    if Initialization:
        # Cria matriz de zeros
        c = np.zeros(shape)
    else:
        # Cria matriz aleatória entre -1 e 1
        # np.random.rand cria valores entre [0, 1)
        # Fórmula: -1 + (2 * rand) -> intervalo [-1, 1)
        c = -1 + 2 * np.random.rand(*shape)
        
    return c

def Qlearning_MultipleChannels(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula o protocolo Q-Learning com múltiplos canais e NOMA.
    
    Args:
        Devices (int): Número de dispositivos.
        Relays (int): Número de relés.
        Channels_Relays (int): Número de canais disponíveis.
        runs (int): Número de rodadas de simulação.
        frames (int): Número de quadros.
        Slots (int): Número de slots de tempo.
        SNR (numpy.ndarray): Matriz de SNR (Devices x Relays x runs).
        N (float): Potência do ruído.
        r (float): Taxa alvo (threshold) para decodificação.
        QTable (numpy.ndarray): Tabela Q (Relays x Slots x Runs x Devices).
        alpha (float): Taxa de aprendizado.
        gamma (float): Fator de desconto.
        
    Returns:
        tuple: (ntput, ndist, ntotal)
    """

    ThroughputRuns = []
    TotalTraffic = np.zeros(runs)
    TotalTrafficDistinct = np.zeros(runs)

    # Pré-alocação para armazenar escolhas
    SlotChoosen = np.zeros((Devices, runs), dtype=int)
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)

    for l in range(frames):
        ThroughputFrame = np.zeros(runs)
        SuccessTransmission = np.zeros((Devices, runs))
        
        # ======================================================================
        # 1. Q-Learning Search (Escolha de Slot e Canal)
        # ======================================================================
        
        # Iterar sobre Dispositivos e Runs para escolher a ação baseada na Q-Table
        # A QTable no MATLAB é (Canal, Slot, Run, Device).
        # Em Python mantemos: (Channels, Slots, Runs, Devices)
        
        # Para otimizar, podemos vetorizar a busca do máximo
        # QTable_CurrentFrame: shape (Channels, Slots, Runs, Devices)
        # Queremos encontrar o índice (canal, slot) do máximo para cada (run, device)
        
        # Abordagem loopada (mais fiel ao original para clareza, mas lenta em Python puro)
        # Vamos tentar vetorizar onde possível, mas manter a lógica de "empate aleatório"
        
        for dd in range(Devices):
            for rr in range(runs):
                # Extrair plano para este dispositivo e run específico
                # Shape: (Channels, Slots)
                QTableDevice = QTable[:, :, rr, dd]
                
                maximum = np.max(QTableDevice)
                
                # Encontrar onde os valores são iguais ao máximo
                # np.where retorna tupla de arrays de índices: (rows, cols) -> (canais, slots)
                y, x = np.where(QTableDevice == maximum)
                
                if len(x) > 1:
                    # Se houver empates, escolhe um aleatoriamente
                    randomChoice = np.random.randint(0, len(x))
                    # y é canal (row), x é slot (col)
                    # Ajuste de índice: MATLAB 1-based para Python 0-based, 
                    # mas aqui salvamos o índice real (0 a N-1) para usar depois
                    ChannelChoosen[dd, rr] = y[randomChoice]
                    SlotChoosen[dd, rr] = x[randomChoice]
                else:
                    ChannelChoosen[dd, rr] = y[0]
                    SlotChoosen[dd, rr] = x[0]

        # Recompensa inicial: -1 (falha por padrão)
        Reward = -1 * np.ones((Devices, runs))

        # ======================================================================
        # 2. Processo de Transmissão (Similar ao Slotted Aloha)
        # ======================================================================
        
        # Loop Slots (0 a Slots-1)
        # Nota: SlotChoosen guarda índices 0 a Slots-1 agora.
        for k in range(Slots):
            for s in range(runs):
                # Encontrar dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]

                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]
                    uniqueChannels = np.unique(TransmittingChannel)

                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)
                        SNR_Device_Channel = SNR_Device[mask_channel, :]
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]

                        # Ordenação Descendente (NOMA SIC)
                        # Ordena ao longo das linhas (axis 0) dentro de cada coluna (relé)
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)
                        
                        # Mapear IDs originais
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                        for rr in range(Relays):
                            SIC_boolean = 0
                            num_users_channel = SNR_Device_Channel.shape[0]

                            for jj in range(num_users_channel):
                                Interference = np.sum(SNR_Device_ord[jj+1:, rr])
                                Signal = SNR_Device_ord[jj, rr]
                                SINR = Signal / (Interference + N)

                                if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                    ThroughputFrame[s] += 1
                                    
                                    device_id = TransmittingDevices_ord[jj, rr]
                                    
                                    # Sucesso! Recompensa positiva e contabilização
                                    Reward[device_id, s] = 1
                                    SuccessTransmission[device_id, s] += 1
                                else:
                                    SIC_boolean = 1

        # ======================================================================
        # 3. Atualização da Q-Table (Bellman Equation)
        # ======================================================================
        
        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]
                sl_idx = SlotChoosen[dd, rr]
                
                # Valor atual na tabela
                QTableDeviceVal = QTable[ch_idx, sl_idx, rr, dd]
                
                # Atualização: (1-alpha)*Q + alpha*(Reward + gamma*Q)
                # Nota: No MATLAB original a fórmula é Q = (1-alpha)*Q + alpha*(R + gamma*Q)
                # Isso é um pouco diferente do Q-Learning padrão (que usa max Q' do próximo estado),
                # mas mantivemos fiel ao código MATLAB fornecido.
                ValorQTable = (1 - alpha) * QTableDeviceVal + alpha * (Reward[dd, rr] + gamma * QTableDeviceVal)
                
                QTable[ch_idx, sl_idx, rr, dd] = ValorQTable

        # Estatísticas
        active_success = np.sum(SuccessTransmission > 0, axis=0)
        ThroughputRuns.append(np.mean(active_success) / Slots)
        
        TotalTrafficDistinct += active_success
        TotalTraffic += np.sum(SuccessTransmission, axis=0)

    # Resultados Finais
    ndist = np.mean(TotalTrafficDistinct) / frames
    ntotal = np.mean(TotalTraffic) / frames

    if ntotal == 0:
        ratio = 0
    else:
        ratio = ndist / ntotal
        
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio
    
    return ntput, ndist, ntotal

def Qlearning_MultipleChannels_NoNOMA(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula o protocolo Q-Learning com múltiplos canais SEM NOMA (apenas Capture Effect).
    
    Args:
        Devices (int): Número de dispositivos.
        Relays (int): Número de relés.
        Channels_Relays (int): Número de canais disponíveis.
        runs (int): Número de rodadas de simulação.
        frames (int): Número de quadros.
        Slots (int): Número de slots de tempo.
        SNR (numpy.ndarray): Matriz de SNR (Devices x Relays x runs).
        N (float): Potência do ruído.
        r (float): Taxa alvo (threshold) para decodificação.
        QTable (numpy.ndarray): Tabela Q (Relays x Slots x Runs x Devices).
        alpha (float): Taxa de aprendizado.
        gamma (float): Fator de desconto.
        
    Returns:
        tuple: (ntput, ndist, ntotal)
    """

    ThroughputRuns = []
    TotalTraffic = np.zeros(runs)
    TotalTrafficDistinct = np.zeros(runs)

    # Arrays para armazenar as escolhas de ação (Canal e Slot)
    SlotChoosen = np.zeros((Devices, runs), dtype=int)
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)

    for l in range(frames):
        ThroughputFrame = np.zeros(runs)
        SuccessTransmission = np.zeros((Devices, runs))
        
        # ======================================================================
        # 1. Q-Learning Search (Escolha de Slot e Canal)
        # ======================================================================
        
        for dd in range(Devices):
            for rr in range(runs):
                # Extrair plano para este dispositivo e run específico
                # QTable shape: (Channels, Slots, Runs, Devices)
                QTableDevice = QTable[:, :, rr, dd]
                
                maximum = np.max(QTableDevice)
                
                # Encontrar índices onde o valor é igual ao máximo
                y, x = np.where(QTableDevice == maximum)
                
                if len(x) > 1:
                    # Empate: escolha aleatória
                    randomChoice = np.random.randint(0, len(x))
                    ChannelChoosen[dd, rr] = y[randomChoice]
                    SlotChoosen[dd, rr] = x[randomChoice]
                else:
                    # Único máximo
                    ChannelChoosen[dd, rr] = y[0]
                    SlotChoosen[dd, rr] = x[0]

        # Inicializa recompensa com penalidade (-1)
        Reward = -1 * np.ones((Devices, runs))

        # ======================================================================
        # 2. Processo de Transmissão
        # ======================================================================
        
        for k in range(Slots):
            for s in range(runs):
                # Encontrar dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]

                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]
                    uniqueChannels = np.unique(TransmittingChannel)

                    for c in uniqueChannels:
                        # Filtros para o canal atual
                        mask_channel = (TransmittingChannel == c)
                        SNR_Device_Channel = SNR_Device[mask_channel, :]
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]
                        
                        # Ordenação Descendente (para identificar o mais forte)
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                        for rr in range(Relays):
                            SIC_boolean = 0
                            num_users_channel = SNR_Device_Channel.shape[0]
                            
                            # ==================================================
                            # Loop NoNOMA: range(1)
                            # Tenta decodificar APENAS o usuário mais forte (índice 0)
                            # ==================================================
                            for jj in range(1): 
                                if num_users_channel > 0: # Garante que há alguém transmitindo
                                    
                                    # Interferência: Soma de todos os outros sinais (do 2º em diante)
                                    Interference = np.sum(SNR_Device_ord[jj+1:, rr])
                                    
                                    Signal = SNR_Device_ord[jj, rr]
                                    SINR = Signal / (Interference + N)

                                    if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                        ThroughputFrame[s] += 1
                                        
                                        device_id = TransmittingDevices_ord[jj, rr]
                                        
                                        # Sucesso! Recompensa positiva
                                        Reward[device_id, s] = 1
                                        SuccessTransmission[device_id, s] += 1
                                    else:
                                        SIC_boolean = 1
                                        
        # ======================================================================
        # 3. Atualização da Q-Table
        # ======================================================================
        
        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]
                sl_idx = SlotChoosen[dd, rr]
                
                # Valor atual na tabela
                QTableDeviceVal = QTable[ch_idx, sl_idx, rr, dd]
                
                # Equação de atualização (baseada no código MATLAB original)
                ValorQTable = (1 - alpha) * QTableDeviceVal + alpha * (Reward[dd, rr] + gamma * QTableDeviceVal)
                
                QTable[ch_idx, sl_idx, rr, dd] = ValorQTable

        # Estatísticas
        active_success = np.sum(SuccessTransmission > 0, axis=0)
        ThroughputRuns.append(np.mean(active_success) / Slots)
        
        TotalTrafficDistinct += active_success
        TotalTraffic += np.sum(SuccessTransmission, axis=0)

    # Resultados Finais
    ndist = np.mean(TotalTrafficDistinct) / frames
    ntotal = np.mean(TotalTraffic) / frames
    
    if ntotal == 0:
        ratio = 0
    else:
        ratio = ndist / ntotal
        
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio
    
    return ntput, ndist, ntotal


def Qlearning_UniqueChannel(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, QTable, alpha, gamma):
    """
    Simula Q-Learning com 'Unique Channel' (Canais/Relés Ortogonais).
    
    Diferente da versão MultipleChannels, aqui assume-se que o dispositivo
    escolhe um canal e o sinal é recebido especificamente através desse canal/relé.
    
    Args:
        Devices (int): Número de dispositivos.
        Relays (int): Número de relés.
        Channels_Relays (int): Número de canais disponíveis.
        runs (int): Número de rodadas.
        frames (int): Número de quadros.
        Slots (int): Número de slots.
        SNR (numpy.ndarray): Matriz de SNR (Devices x Relays x runs).
        N (float): Potência do ruído.
        r (float): Taxa alvo.
        QTable (numpy.ndarray): Tabela Q (Channels x Slots x Runs x Devices).
        alpha (float): Taxa de aprendizado.
        gamma (float): Fator de desconto.
        
    Returns:
        tuple: (ntput, ndist, ntotal)
    """

    ThroughputRuns = []
    TotalTraffic = np.zeros(runs)
    TotalTrafficDistinct = np.zeros(runs)

    # Pré-alocação das escolhas
    SlotChoosen = np.zeros((Devices, runs), dtype=int)
    ChannelChoosen = np.zeros((Devices, runs), dtype=int)

    for l in range(frames):
        ThroughputFrame = np.zeros(runs)
        SuccessTransmission = np.zeros((Devices, runs))
        
        # ======================================================================
        # 1. Q-Learning Search (Escolha de Ação)
        # ======================================================================
        for dd in range(Devices):
            for rr in range(runs):
                # QTable shape: (Channels, Slots, Runs, Devices)
                QTableDevice = QTable[:, :, rr, dd]
                
                maximum = np.max(QTableDevice)
                
                # Encontrar índices (canal, slot) do máximo
                y, x = np.where(QTableDevice == maximum)
                
                if len(x) > 1:
                    # Empate: escolha aleatória
                    rand_idx = np.random.randint(0, len(x))
                    ChannelChoosen[dd, rr] = y[rand_idx]
                    SlotChoosen[dd, rr] = x[rand_idx]
                else:
                    # Único
                    ChannelChoosen[dd, rr] = y[0]
                    SlotChoosen[dd, rr] = x[0]

        # Inicializa recompensas com penalidade
        Reward = -1 * np.ones((Devices, runs))

        # ======================================================================
        # 2. Processo de Transmissão
        # ======================================================================
        for k in range(Slots):
            for s in range(runs):
                # Dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]

                if len(TransmittingDevices) >= 1:
                    # --- Extração de SNR Específico (Unique Channel Logic) ---
                    # MATLAB: ind = sub2ind(size(SNR), TransmittingDevices, TransmittingChannel, ...)
                    # Python: Indexação avançada (Fancy indexing)
                    # SNR shape: (Devices, Relays, Runs)
                    # Assumimos que o índice do canal escolhido corresponde ao índice do relé na matriz SNR.
                    
                    # Extrai apenas o SNR do canal/relé que o dispositivo escolheu
                    SNR_Device = SNR[TransmittingDevices, TransmittingChannel, s]
                    
                    uniqueChannels = np.unique(TransmittingChannel)

                    for c in uniqueChannels:
                        # Filtra dispositivos neste canal
                        mask_channel = (TransmittingChannel == c)
                        
                        # SNRs dos dispositivos neste canal
                        SNR_Device_Channel = SNR_Device[mask_channel]
                        
                        # IDs dos dispositivos
                        TransmittingDevices_Channel = TransmittingDevices[mask_channel]

                        # Ordenação Descendente para SIC
                        # Note que aqui SNR_Device_Channel é 1D (apenas os valores deste canal)
                        sort_indexes = np.argsort(SNR_Device_Channel)[::-1]
                        
                        SNR_Device_ord = SNR_Device_Channel[sort_indexes]
                        TransmittingDevices_ord = TransmittingDevices_Channel[sort_indexes]

                        # Loop SIC
                        SIC_boolean = 0
                        num_users = len(SNR_Device_ord)
                        
                        for jj in range(num_users):
                            # Interferência: Soma de todos os usuários mais fracos (do próximo em diante)
                            Interference = np.sum(SNR_Device_ord[jj+1:])
                            
                            Signal = SNR_Device_ord[jj]
                            SINR = Signal / (Interference + N)

                            if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                ThroughputFrame[s] += 1
                                
                                dev_id = TransmittingDevices_ord[jj]
                                Reward[dev_id, s] = 1
                                SuccessTransmission[dev_id, s] += 1
                            else:
                                # Se falhar na decodificação do mais forte, SIC para
                                SIC_boolean = 1

        # ======================================================================
        # 3. Atualização da Q-Table
        # ======================================================================
        for dd in range(Devices):
            for rr in range(runs):
                ch_idx = ChannelChoosen[dd, rr]
                sl_idx = SlotChoosen[dd, rr]
                
                current_q = QTable[ch_idx, sl_idx, rr, dd]
                
                # Atualização baseada na equação do MATLAB
                new_q = (1 - alpha) * current_q + alpha * (Reward[dd, rr] + gamma * current_q)
                
                QTable[ch_idx, sl_idx, rr, dd] = new_q

        # Estatísticas
        active_success = np.sum(SuccessTransmission > 0, axis=0)
        ThroughputRuns.append(np.mean(active_success) / Slots)
        
        TotalTrafficDistinct += active_success
        TotalTraffic += np.sum(SuccessTransmission, axis=0)

    # Resultados Finais
    ndist = np.mean(TotalTrafficDistinct) / frames
    ntotal = np.mean(TotalTraffic) / frames
    
    if ntotal == 0:
        ratio = 0
    else:
        ratio = ndist / ntotal
        
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio
    
    return ntput, ndist, ntotal