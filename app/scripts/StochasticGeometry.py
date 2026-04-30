import numpy as np  # Importa NumPy para operações numéricas com arrays

def StochasticGeometry(Devices, Relays, Radius, runs, return_coords=False):
    """
    Gera posições aleatórias de dispositivos e relays dentro de uma célula circular
    e calcula a matriz de distâncias euclidianas entre eles.
    
    Args:
        Devices (int): Número de dispositivos IoT a posicionar na célula.
        Relays (int): Número de nós relays a posicionar na célula.
        Radius (float): Raio da célula circular (metros).
        runs (int): Número de rodadas independentes de simulação Monte Carlo.
        return_coords (bool): Se True, retorna também as coordenadas cartesianas
                              (x_dev, y_dev, x_rel, y_rel) além da matriz de distâncias.

    Returns:
        numpy.ndarray: Matriz 3D de distâncias com shape (Devices, Relays, runs),
                       em que cada entrada [d, r, run] é a distância em metros
                       entre o dispositivo d e o relay r na rodada 'run'.
        Se return_coords=True, retorna tupla (distances, x_dev, y_dev, x_rel, y_rel)
        onde x_dev/y_dev têm shape (Devices, runs) e x_rel/y_rel têm shape (Relays, runs).
    """

    # --- Passo 1: Gerar posições dos relays aleatoriamente dentro da célula circular ---
    # Usa sqrt(uniforme) para garantir distribuição uniforme sobre a área circular
    r_rel = Radius * np.sqrt(np.random.rand(Relays, runs))    # Distância radial de cada relay por rodada
    th_rel = 2 * np.pi * np.random.rand(Relays, runs)         # Ângulo polar (0 a 2π) de cada relay por rodada
    x_rel = r_rel * np.cos(th_rel)                            # Converte coordenada polar para X cartesiano
    y_rel = r_rel * np.sin(th_rel)                            # Converte coordenada polar para Y cartesiano

    # --- Passo 2: Gerar posições dos dispositivos aleatoriamente dentro da célula circular ---
    r_dev = Radius * np.sqrt(np.random.rand(Devices, runs))   # Distância radial de cada dispositivo por rodada
    th_dev = 2 * np.pi * np.random.rand(Devices, runs)        # Ângulo polar (0 a 2π) de cada dispositivo por rodada
    x_dev = r_dev * np.cos(th_dev)                            # Converte coordenada polar para X cartesiano
    y_dev = r_dev * np.sin(th_dev)                            # Converte coordenada polar para Y cartesiano

    # --- Passo 3: Calcular distâncias par-a-par via broadcasting do NumPy ---
    # Expande dimensões: dispositivos -> (Devices, 1, runs), relays -> (1, Relays, runs)
    # O broadcasting produz automaticamente shape (Devices, Relays, runs)
    diff_x = x_dev[:, np.newaxis, :] - x_rel[np.newaxis, :, :]  # Diferença no eixo X entre cada par dispositivo-relay
    diff_y = y_dev[:, np.newaxis, :] - y_rel[np.newaxis, :, :]  # Diferença no eixo Y entre cada par dispositivo-relay

    # Calcula a distância Euclidiana para cada tripla (dispositivo, relay, rodada)
    c = np.sqrt(diff_x**2 + diff_y**2)

    if return_coords:
        return c, x_dev, y_dev, x_rel, y_rel  # Retorna distâncias + coordenadas cartesianas
    return c  # Retorna apenas a matriz 3D de distâncias (comportamento padrão)