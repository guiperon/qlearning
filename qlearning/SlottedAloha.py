import numpy as np

def SlottedAloha_MultipleChannels(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r):
    """
    Simula o protocolo Slotted Aloha com múltiplos canais e NOMA.
    
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
        
    Returns:
        tuple: (ntput, ndist, ntotal)
    """
    
    ThroughputRuns = []
    TotalTraffic = np.zeros(runs)
    TotalTrafficDistinct = np.zeros(runs)

    # Loop Frames
    for l in range(frames):
        SuccessTransmission = np.zeros((Devices, runs))
        ThroughputFrame = np.zeros(runs)
        
        # Escolha aleatória de Slots e Canais
        # MATLAB: randi(Slots) -> inteiros de 1 a Slots
        # Python: np.random.randint(1, Slots + 1)
        SlotChoosen = np.random.randint(1, Slots + 1, size=(Devices, runs))
        ChannelChoosen = np.random.randint(1, Channels_Relays + 1, size=(Devices, runs))

        # Loop Slots
        for k in range(1, Slots + 1):
            # Loop Runs (Monte Carlo)
            for s in range(runs):
                # Encontrar dispositivos transmitindo no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]

                if len(TransmittingDevices) >= 1:
                    # Extrair SNR para os dispositivos ativos nesta rodada
                    # SNR shape: (Devices, Relays, runs)
                    SNR_Device = SNR[TransmittingDevices, :, s]
                    
                    uniqueChannels = np.unique(TransmittingChannel)

                    for c in uniqueChannels:
                        # Filtros para o canal atual
                        mask_channel = (TransmittingChannel == c)
                        SNR_Device_Channel = SNR_Device[mask_channel, :]
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]
                        
                        # Ordenação Descendente (NOMA SIC)
                        # MATLAB: sort(..., 1, 'descend') ordena cada coluna (relé) independentemente
                        # Python: argsort retorna índices para ordenar
                        # axis=0 ordena ao longo das linhas (dentro de cada coluna)
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1] # [::-1] inverte para descendente
                        
                        # Reordenar SNR e IDs dos dispositivos baseados na força do sinal para cada relé
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)
                        
                        # Mapear os índices ordenados de volta para os IDs originais dos dispositivos
                        # TransmittingDevices_channel é (N_devs,). sort_indexes é (N_devs, Relays).
                        # O resultado deve ser (N_devs, Relays), indicando qual dispositivo é o 1º, 2º... em cada relé.
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                        # Loop Relés
                        # uniqueRelays no MATLAB era 1:1:Relays. Em Python iteramos range(Relays).
                        for rr in range(Relays):
                            SIC_boolean = 0
                            num_users_channel = SNR_Device_Channel.shape[0]
                            
                            # SIC Loop (Interference Cancellation)
                            for jj in range(num_users_channel):
                                # Interferência: Soma dos sinais mais fracos (abaixo do atual jj)
                                # MATLAB: sum(SNR_Device_ord((jj+1):end, uniqueRelays(rr)))
                                # Python: slice [jj+1:, rr]
                                Interference = np.sum(SNR_Device_ord[jj+1:, rr])
                                
                                Signal = SNR_Device_ord[jj, rr]
                                SINR = Signal / (Interference + N)

                                if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                    ThroughputFrame[s] += 1
                                    
                                    # Identificar o dispositivo que teve sucesso
                                    device_id = TransmittingDevices_ord[jj, rr]
                                    SuccessTransmission[device_id, s] += 1
                                else:
                                    # Se falhar em decodificar o mais forte, SIC para para os subsequentes
                                    SIC_boolean = 1
        
        # Estatísticas por Frame
        # MATLAB: mean(sum(SuccessTransmission>0))/Slots
        active_success = np.sum(SuccessTransmission > 0, axis=0) # Soma ao longo dos dispositivos
        ThroughputRuns.append(np.mean(active_success) / Slots)
        
        TotalTrafficDistinct += active_success
        TotalTraffic += np.sum(SuccessTransmission, axis=0)

    # Cálculos Finais
    ndist = np.mean(TotalTrafficDistinct) / frames
    ntotal = np.mean(TotalTraffic) / frames
    
    # Evitar divisão por zero se não houver tráfego
    if ntotal == 0:
        ratio = 0
    else:
        ratio = ndist / ntotal
        
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio
    
    return ntput, ndist, ntotal




def SlottedAloha_MultipleChannels_NoNOMA(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r):
    """
    Simula o protocolo Slotted Aloha com múltiplos canais SEM NOMA (apenas Capture Effect).
    
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
        
    Returns:
        tuple: (ntput, ndist, ntotal)
    """
    
    ThroughputRuns = []
    TotalTraffic = np.zeros(runs)
    TotalTrafficDistinct = np.zeros(runs)

    # Loop Frames
    for l in range(frames):
        SuccessTransmission = np.zeros((Devices, runs))
        ThroughputFrame = np.zeros(runs)
        
        # Escolha aleatória de Slots e Canais
        # (1 a Slots+1) para manter compatibilidade lógica, embora python seja 0-indexed
        SlotChoosen = np.random.randint(1, Slots + 1, size=(Devices, runs))
        ChannelChoosen = np.random.randint(1, Channels_Relays + 1, size=(Devices, runs))

        # Loop Slots
        for k in range(1, Slots + 1):
            # Loop Runs
            for s in range(runs):
                # Identificar dispositivos transmitindo no slot k
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]

                if len(TransmittingDevices) >= 1:
                    # Extrair SNR (Devices x Relays) para a rodada 's'
                    SNR_Device = SNR[TransmittingDevices, :, s]
                    
                    uniqueChannels = np.unique(TransmittingChannel)

                    for c in uniqueChannels:
                        # Filtros para o canal atual
                        mask_channel = (TransmittingChannel == c)
                        SNR_Device_Channel = SNR_Device[mask_channel, :]
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]
                        
                        # Ordenação Descendente (Similar ao NOMA, para achar o mais forte)
                        # Axis 0 = ordena as linhas (dispositivos) dentro de cada coluna (relé)
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]
                        
                        # Reordenar SNR e IDs
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)
                        
                        # Mapear IDs originais para a matriz ordenada (Devices x Relays)
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                        # Loop Relés
                        for rr in range(Relays):
                            SIC_boolean = 0
                            
                            # Loop jj=1:1 (NoNOMA / Capture Effect)
                            # Tenta decodificar APENAS o primeiro (mais forte)
                            for jj in range(1): 
                                if SNR_Device_ord.shape[0] > jj: # Verifica se existe pelo menos 1 usuário
                                    
                                    # Interferência: Soma de todos os outros sinais (do 2º em diante)
                                    Interference = np.sum(SNR_Device_ord[jj+1:, rr])
                                    
                                    Signal = SNR_Device_ord[jj, rr]
                                    SINR = Signal / (Interference + N)

                                    # Verifica condição de sucesso e se o SIC não falhou (irrelevante aqui pois só roda 1x, mas mantido pela lógica)
                                    if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                        ThroughputFrame[s] += 1
                                        
                                        # Contabilizar sucesso para o dispositivo específico
                                        device_id = TransmittingDevices_ord[jj, rr]
                                        SuccessTransmission[device_id, s] += 1
                                    else:
                                        SIC_boolean = 1
        
        # Cálculo de estatísticas do Frame
        # Soma sucessos únicos por run (>0 transforma contagem em booleano)
        active_success = np.sum(SuccessTransmission > 0, axis=0)
        
        ThroughputRuns.append(np.mean(active_success) / Slots)
        TotalTrafficDistinct += active_success
        TotalTraffic += np.sum(SuccessTransmission, axis=0)

    # Médias finais
    ndist = np.mean(TotalTrafficDistinct) / frames
    ntotal = np.mean(TotalTraffic) / frames
    
    if ntotal == 0:
        ratio = 0
    else:
        ratio = ndist / ntotal
        
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio
    
    return ntput, ndist, ntotal