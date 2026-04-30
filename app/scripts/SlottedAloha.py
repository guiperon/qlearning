import numpy as np  # Importa NumPy para operações numéricas com arrays

def SlottedAloha_MultipleChannels(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r):
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

                        # Ordena dispositivos do mais forte ao mais fraco (descendente) para SIC-NOMA
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]                          # Índices de ordenação descendente por relay
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)        # SNR reordenado (mais forte primeiro)
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]                   # IDs reordenados conforme a força do sinal

                        # --- Processa cada relay independentemente ---
                        for rr in range(Relays):
                            SIC_boolean = 0                                    # Flag de falha SIC: 0 = SIC ativo, 1 = SIC falhou
                            num_users_channel = SNR_Device_Channel.shape[0]    # Número de usuários colidindo neste canal

                            # --- Loop SIC: tenta decodificar do mais forte ao mais fraco ---
                            for jj in range(num_users_channel):
                                Interference = np.sum(SNR_Device_ord[jj+1:, rr])  # Interferência = soma dos sinais mais fracos restantes
                                Signal = SNR_Device_ord[jj, rr]                    # Potência do sinal do usuário atual
                                SINR = Signal / (Interference + N)                 # Calcula SINR (sinal sobre interferência + ruído)

                                # Verifica se a taxa alcançável supera o limiar e se o SIC ainda não falhou
                                if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                    ThroughputFrame[s] += 1                                    # Incrementa throughput do frame
                                    device_id = TransmittingDevices_ord[jj, rr]                # Obtém o ID original do dispositivo decodificado
                                    SuccessTransmission[device_id, s] += 1                     # Registra sucesso para este dispositivo
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




def SlottedAloha_MultipleChannels_NoNOMA(Devices, Relays, Channels_Relays, runs, frames, Slots, SNR, N, r):
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

                        # Ordena do mais forte ao mais fraco para identificar o dispositivo dominante
                        sort_indexes = np.argsort(SNR_Device_Channel, axis=0)[::-1]                          # Índices de ordenação descendente
                        SNR_Device_ord = np.take_along_axis(SNR_Device_Channel, sort_indexes, axis=0)        # SNR reordenado
                        TransmittingDevices_ord = TransmittingDevices_channel[sort_indexes]                   # IDs reordenados

                        # --- Processa cada relay ---
                        for rr in range(Relays):
                            SIC_boolean = 0  # Flag de falha (mantida por compatibilidade com a estrutura NOMA)

                            # Sem NOMA: tenta decodificar APENAS o usuário mais forte (range(1) = só índice 0)
                            for jj in range(1):
                                if SNR_Device_ord.shape[0] > jj:  # Garante que existe pelo menos 1 usuário
                                    Interference = np.sum(SNR_Device_ord[jj+1:, rr])  # Interferência dos demais dispositivos
                                    Signal = SNR_Device_ord[jj, rr]                    # Sinal do dispositivo mais forte
                                    SINR = Signal / (Interference + N)                 # Calcula SINR

                                    # Verifica se a taxa alcançável supera o limiar
                                    if (np.log2(1 + SINR) >= r) and (SIC_boolean == 0):
                                        ThroughputFrame[s] += 1                            # Incrementa throughput
                                        device_id = TransmittingDevices_ord[jj, rr]        # ID do dispositivo decodificado
                                        SuccessTransmission[device_id, s] += 1             # Registra sucesso
                                    else:
                                        SIC_boolean = 1  # Marca falha na decodificação

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