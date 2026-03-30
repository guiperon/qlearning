import numpy as np

def StochasticGeometry(Devices, Relays, Radius, runs):
    """
    Versão vetorizada para alta performance.
    """
    # 1. Gerar posições de TODOS os Relés de uma vez
    # Shape: (Relays, runs)
    r_rel = Radius * np.sqrt(np.random.rand(Relays, runs))
    th_rel = 2 * np.pi * np.random.rand(Relays, runs)
    x_rel = r_rel * np.cos(th_rel)
    y_rel = r_rel * np.sin(th_rel)

    # 2. Gerar posições de TODOS os Dispositivos de uma vez
    # Shape: (Devices, runs)
    r_dev = Radius * np.sqrt(np.random.rand(Devices, runs))
    th_dev = 2 * np.pi * np.random.rand(Devices, runs)
    x_dev = r_dev * np.cos(th_dev)
    y_dev = r_dev * np.sin(th_dev)

    # 3. Calcular distâncias usando Broadcasting (evita loops)
    # Expandimos as dimensões para: (Devices, 1, runs) e (1, Relays, runs)
    # O resultado será (Devices, Relays, runs)
    
    # Diferença nas coordenadas X e Y
    diff_x = x_dev[:, np.newaxis, :] - x_rel[np.newaxis, :, :]
    diff_y = y_dev[:, np.newaxis, :] - y_rel[np.newaxis, :, :]
    
    # Distância Euclidiana
    c = np.sqrt(diff_x**2 + diff_y**2)
    
    return c