def measure_speedup(cpu_time: float, gpu_time: float, num_units: int):
    """
    Calcula el speedup y la eficiencia de la ejecución GPU vs CPU.
    """
    if gpu_time == 0:
        return float('inf'), 1.0
    speedup = cpu_time / gpu_time
    efficiency = speedup / num_units
    return speedup, efficiency