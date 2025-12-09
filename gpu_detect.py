import torch
import tensorflow as tf
def test_pytorch():
   print("Testing PyTorch...")
   if torch.cuda.is_available():
       print(f"PyTorch: NVIDIA GPU detected! Device: {torch.cuda.get_device_name(0)}")
   else:
       print("PyTorch: No NVIDIA GPU detected.")
def test_tensorflow():
   print("\nTesting TensorFlow...")
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       print(f"TensorFlow: {len(gpus)} NVIDIA GPU(s) detected!")
       for gpu in gpus:
           print(f" - {gpu.name}")
   else:
       print("TensorFlow: No NVIDIA GPU detected.")
if __name__ == "__main__":
   test_pytorch()
   test_tensorflow()