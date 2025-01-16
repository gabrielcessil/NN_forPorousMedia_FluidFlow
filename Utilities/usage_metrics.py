import psutil
import torch

def get_ProcessingTime(nOperations, flops, nLoops=1): return nLoops*nOperations/flops

def get_MemoryUsage_MB(nParameters, nBytes): return nParameters*nBytes/ 1048576

def check_memory(device="cpu"):
    """
    Verifica a memória disponível e usada no dispositivo especificado.
    
    Args:
        device (str): 'cpu' ou 'cuda'. Por padrão, 'cpu'.

    Returns:
        dict: Informações sobre a memória (total, usada, livre).
    """
    if device == "cuda" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory
        
        print(f"\n\nDispositivo: GPU")
        print(f"Memória Total: {total_memory / (1024**2):.2f} MB")
        print(f"Memória Reservada: {reserved_memory / (1024**2):.2f} MB")
        print(f"Memória Alocada: {allocated_memory / (1024**2):.2f} MB")
        print(f"Memória Livre: {free_memory / (1024**2):.2f} MB")
        return {"total": total_memory, "used": allocated_memory, "free": free_memory}
    else:
        memory = psutil.virtual_memory()
        total_memory = memory.total
        used_memory = memory.used
        free_memory = memory.available

        print(f"\n\nDispositivo: CPU")
        print(f"Memória Total: {total_memory / (1024**2):.2f} MB")
        print(f"Memória Usada: {used_memory / (1024**2):.2f} MB")
        print(f"Memória Livre: {free_memory / (1024**2):.2f} MB")
        return {"total": total_memory, "used": used_memory, "free": free_memory}

import gc

def clear_gpu_memory():
    """Clears unused GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_cpu_memory():
    """Free up memory used by CPU tensors explicitly."""
    gc.collect()

def delete_model(model):
    """Removes the PyTorch model and frees associated memory."""
    # Delete the model reference
    del model

    # Clear GPU memory if the model was on the GPU
    clear_gpu_memory()

    # Clear CPU memory using garbage collection
    clear_cpu_memory()

    print("PyTorch model and associated memory freed.")