import numpy as np  # Importa NumPy para operações numéricas com arrays

def SlottedAloha_MultipleChannels(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, ClusterAssignment=None):
    """
    Simula o protocolo Slotted Aloha com múltiplos canais e NOMA (SIC).
    Cada dispositivo escolhe aleatoriamente um slot e um canal para transmitir.
    A decodificação usa Cancelamento Sucessivo de Interferência (SIC).

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
        ClusterAssignment (numpy.ndarray, optional): Índice do relay atribuído a cada dispositivo
            por rodada, shape (Devices, runs). Se None, todos os relays processam todos os
            dispositivos (comportamento original). Se fornecido, cada relay processa apenas os
            dispositivos a ele atribuídos (clusterização distribuída por RSSI).

    Returns:
        tuple: (ntput, ndist, ntotal) — throughput normalizado, tráfego distinto médio, tráfego total médio.
    """

    ThroughputRuns = []                        # Lista para armazenar throughput médio de cada frame
    TotalTraffic = np.zeros(runs)              # Acumulador de tráfego total (incluindo duplicatas) por rodada
    TotalTrafficDistinct = np.zeros(runs)      # Acumulador de tráfego distinto (dispositivos únicos) por rodada

    # --- Iteração sobre cada quadro de transmissão ---
    for l in range(frames):
        SuccessTransmission = np.zeros((Devices, runs))  # Matriz de sucessos: conta quantas vezes cada dispositivo foi decodificado
        ThroughputFrame = np.zeros(runs)                  # Throughput acumulado neste frame por rodada

        # Cada dispositivo escolhe aleatoriamente um slot (1 a Slots) e um canal (1 a Channels_Relays)
        SlotChoosen = np.random.randint(1, Slots + 1, size=(Devices, runs))              # Slot escolhido por cada dispositivo
        ChannelChoosen = np.random.randint(1, Channels_Relays + 1, size=(Devices, runs)) # Canal escolhido por cada dispositivo

        # --- Iteração sobre cada slot de tempo ---
        for k in range(1, Slots + 1):
            # --- Iteração Monte Carlo sobre cada rodada ---
            for s in range(runs):
                # Identifica quais dispositivos escolheram transmitir no slot k, rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]       # Índices dos dispositivos ativos
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]     # Canal escolhido por cada dispositivo ativo

                # Verifica se há pelo menos um dispositivo transmitindo neste slot
                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]  # Extrai SNR dos dispositivos ativos para todos os relays
                    uniqueChannels = np.unique(TransmittingChannel)  # Lista de canais distintos em uso neste slot

                    # --- Processa cada canal separadamente ---
                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)                          # Máscara booleana: dispositivos neste canal
                        SNR_Device_Channel = SNR_Device[mask_channel, :]                   # SNR dos dispositivos neste canal específico
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]     # IDs dos dispositivos neste canal

                        if ClusterAssignment is None:
                            # --- Sem clusterização: todos os relays processam todos os dispositivos ---
                            sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]                          # Índices de ordenação descendente por relay
                            SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)        # SNR reordenado (mais forte primeiro)
                            TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]                   # IDs reordenados conforme a força do sinal

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
                                        SuccessTransmission[device_id, s] += 1
                                    else:
                                        SIC_boolean = 1
                        else:
                            # --- Com clusterização RSSI: cada relay processa apenas seus dispositivos atribuídos ---
                            DeviceRelay_channel = ClusterAssignment[TransmittingDevices_channel, s]  # Relay atribuído a cada dispositivo ativo neste canal

                            for rr in range(Relays):
                                mask_relay = (DeviceRelay_channel == rr)       # Filtra dispositivos atribuídos ao relay rr
                                if not np.any(mask_relay):
                                    continue                                    # Nenhum dispositivo atribuído a este relay neste slot/canal

                                SNR_Relay = SNR_Device_Channel[mask_relay, :]              # SNR dos dispositivos do cluster rr
                                Devices_relay = TransmittingDevices_channel[mask_relay]    # IDs dos dispositivos do cluster rr

                                # Ordena dispositivos do mais forte ao mais fraco no relay rr (critério RSSI)
                                sort_idx = np.argsort(SNR_Relay[:, rr])[::-1]
                                SNR_Relay_ord = SNR_Relay[sort_idx, :]
                                Devices_relay_ord = Devices_relay[sort_idx]

                                # --- Loop SIC: decodifica do mais forte ao mais fraco ---
                                SIC_boolean = 0
                                for jj in range(len(Devices_relay_ord)):
                                    Interference = np.sum(SNR_Relay_ord[jj+1:, rr])  # Interferência dos sinais mais fracos restantes
                                    Signal = SNR_Relay_ord[jj, rr]                    # Potência do sinal do usuário atual
                                    SINR = Signal / (Interference + N)

                                    if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                        ThroughputFrame[s] += 1
                                        SuccessTransmission[Devices_relay_ord[jj], s] += 1
                                    else:
                                        SIC_boolean = 1  # Marca falha no SIC — para de decodificar os subsequentes

        # --- Estatísticas ao final de cada frame ---
        active_success = np.sum(SuccessTransmission > 0, axis=0)      # Conta dispositivos únicos com pelo menos 1 sucesso por rodada
        ThroughputRuns.append(np.mean(active_success) / Slots)        # Throughput médio normalizado por slots
        TotalTrafficDistinct += active_success                         # Acumula tráfego distinto ao longo dos frames
        TotalTraffic += np.sum(SuccessTransmission, axis=0)           # Acumula tráfego total (com duplicatas)

    # --- Cálculos finais de desempenho ---
    ndist = np.mean(TotalTrafficDistinct) / frames    # Média de dispositivos distintos por frame
    ntotal = np.mean(TotalTraffic) / frames           # Média de transmissões totais por frame

    # Evita divisão por zero caso não haja tráfego
    if ntotal == 0:
        ratio = 0           # Razão nula se não houve tráfego
    else:
        ratio = ndist / ntotal  # Razão entre tráfego distinto e total (proporção de mensagens úteis)

    # Throughput normalizado final: pondera pela taxa alvo, número de canais e razão de utilidade
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio

    return ntput, ndist, ntotal  # Retorna throughput normalizado, tráfego distinto e tráfego total




def SlottedAloha_MultipleChannels_NoNOMA(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r, ClusterAssignment=None):
    """
    Simula o protocolo Slotted Aloha com múltiplos canais SEM NOMA (apenas Efeito Captura).
    Diferente da versão com NOMA, aqui apenas o dispositivo com sinal mais forte é decodificado.

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
        ClusterAssignment (numpy.ndarray, optional): Índice do relay atribuído a cada dispositivo
            por rodada, shape (Devices, runs). Se None, comportamento original (sem clusterização).
            Se fornecido, cada relay processa apenas os dispositivos a ele atribuídos.

    Returns:
        tuple: (ntput, ndist, ntotal) — throughput normalizado, tráfego distinto médio, tráfego total médio.
    """

    ThroughputRuns = []                        # Lista para armazenar throughput médio de cada frame
    TotalTraffic = np.zeros(runs)              # Acumulador de tráfego total por rodada
    TotalTrafficDistinct = np.zeros(runs)      # Acumulador de tráfego distinto por rodada

    # --- Iteração sobre cada quadro de transmissão ---
    for l in range(frames):
        SuccessTransmission = np.zeros((Devices, runs))  # Matriz de sucessos por dispositivo e rodada
        ThroughputFrame = np.zeros(runs)                  # Throughput acumulado neste frame

        # Cada dispositivo escolhe aleatoriamente um slot e um canal (indexação 1-based)
        SlotChoosen = np.random.randint(1, Slots + 1, size=(Devices, runs))              # Slot escolhido por cada dispositivo
        ChannelChoosen = np.random.randint(1, Channels_Relays + 1, size=(Devices, runs)) # Canal escolhido por cada dispositivo

        # --- Iteração sobre cada slot de tempo ---
        for k in range(1, Slots + 1):
            # --- Iteração Monte Carlo sobre cada rodada ---
            for s in range(runs):
                # Identifica dispositivos transmitindo no slot k da rodada s
                TransmittingDevices = np.where(SlotChoosen[:, s] == k)[0]       # Índices dos dispositivos ativos
                TransmittingChannel = ChannelChoosen[TransmittingDevices, s]     # Canal de cada dispositivo ativo

                # Verifica se há dispositivos transmitindo
                if len(TransmittingDevices) >= 1:
                    SNR_Device = SNR[TransmittingDevices, :, s]       # SNR dos dispositivos ativos para todos os relays
                    uniqueChannels = np.unique(TransmittingChannel)   # Canais distintos em uso neste slot

                    # --- Processa cada canal separadamente ---
                    for c in uniqueChannels:
                        mask_channel = (TransmittingChannel == c)                          # Máscara: dispositivos neste canal
                        SNR_Device_Channel = SNR_Device[mask_channel, :]                   # SNR dos dispositivos neste canal
                        TransmittingDevices_channel = TransmittingDevices[mask_channel]     # IDs dos dispositivos neste canal

                        if ClusterAssignment is None:
                            # --- Sem clusterização: todos os relays processam todos os dispositivos ---
                            sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]
                            SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)
                            TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]

                            for rr in range(Relays):
                                SIC_boolean = 0

                                for jj in range(1):
                                    if SNR_Device_ord.shape[0] > jj:
                                        Interference = np.sum(SNR_Device_ord[jj+1:, rr])
                                        Signal = SNR_Device_ord[jj, rr]
                                        SINR = Signal / (Interference + N)

                                        if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                            ThroughputFrame[s] += 1
                                            device_id = TransmittingDevices_ord[jj, rr]
                                            SuccessTransmission[device_id, s] += 1
                                        else:
                                            SIC_boolean = 1
                        else:
                            # --- Com clusterização RSSI: cada relay processa apenas seus dispositivos atribuídos ---
                            DeviceRelay_channel = ClusterAssignment[TransmittingDevices_channel, s]

                            for rr in range(Relays):
                                mask_relay = (DeviceRelay_channel == rr)
                                if not np.any(mask_relay):
                                    continue

                                SNR_Relay = SNR_Device_Channel[mask_relay, :]
                                Devices_relay = TransmittingDevices_channel[mask_relay]

                                # Ordena por SNR no relay rr e seleciona apenas o mais forte (sem NOMA)
                                sort_idx = np.argsort(SNR_Relay[:, rr])[::-1]
                                SNR_Relay_ord = SNR_Relay[sort_idx, :]
                                Devices_relay_ord = Devices_relay[sort_idx]

                                Interference = np.sum(SNR_Relay_ord[1:, rr])  # Interferência dos demais dispositivos do cluster
                                Signal = SNR_Relay_ord[0, rr]
                                SINR = Signal / (Interference + N)

                                if np.log2(1 + SINR) >= r:
                                    ThroughputFrame[s] += 1
                                    SuccessTransmission[Devices_relay_ord[0], s] += 1

        # --- Estatísticas ao final de cada frame ---
        active_success = np.sum(SuccessTransmission > 0, axis=0)      # Dispositivos únicos com sucesso por rodada
        ThroughputRuns.append(np.mean(active_success) / Slots)        # Throughput normalizado médio
        TotalTrafficDistinct += active_success                         # Acumula tráfego distinto
        TotalTraffic += np.sum(SuccessTransmission, axis=0)           # Acumula tráfego total

    # --- Cálculos finais de desempenho ---
    ndist = np.mean(TotalTrafficDistinct) / frames    # Média de dispositivos distintos decodificados por frame
    ntotal = np.mean(TotalTraffic) / frames           # Média de decodificações totais por frame

    # Evita divisão por zero
    if ntotal == 0:
        ratio = 0           # Sem tráfego, razão é zero
    else:
        ratio = ndist / ntotal  # Proporção de mensagens úteis (distintas / total)

    # Throughput normalizado final
    ntput = (r / Channels_Relays) * np.mean(ThroughputRuns) * ratio

    return ntput, ndist, ntotal  # Retorna throughput normalizado, tráfego distinto e total