import torch
import tensorflow as tf

print("PyTorch GPU Check:")
print(torch.cuda.is_available())  # True = GPU OK
print(torch.cuda.get_device_name(0))


print("\nTensorFlow GPU Check:")
print(tf.config.list_physical_devices('GPU'))
