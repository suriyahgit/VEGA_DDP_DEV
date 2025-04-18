import torch

def test_gpu(gpu_id):
    # Set the device to the GPU id
    device = torch.device(f"cuda:{gpu_id}")
    
    # Print the GPU being tested
    print(f"Testing GPU {gpu_id}...")
    
    # Allocate a tensor on the GPU
    tensor = torch.rand((10000, 10000), device=device)
    
    # Perform a simple operation: Matrix multiplication
    result = torch.mm(tensor, tensor)
    
    # Print the result to ensure the operation was successful
    print(f"GPU {gpu_id} test successful!\n")

def main():
    # Get the number of GPUs available on the system
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs found!")
        return
    
    # Test each GPU
    for gpu_id in range(num_gpus):
        test_gpu(gpu_id)

if __name__ == "__main__":
    main()

