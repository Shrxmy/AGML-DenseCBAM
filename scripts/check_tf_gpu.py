import os
import platform
import subprocess
import sys

print('Python:', sys.version.replace('\n', ' '))
print('Platform:', platform.platform())
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))

print('\n=== nvidia-smi ===')
try:
    result = subprocess.run(['nvidia-smi'], text=True, capture_output=True, timeout=15)
    print(result.stdout if result.stdout else result.stderr)
    print('nvidia-smi exit code:', result.returncode)
except Exception as exc:
    print('nvidia-smi failed:', repr(exc))

print('\n=== TensorFlow ===')
try:
    import tensorflow as tf
    print('TensorFlow:', tf.__version__)
    print('Built with CUDA:', tf.test.is_built_with_cuda())
    print('Physical devices:', tf.config.list_physical_devices())
    print('GPU devices:', tf.config.list_physical_devices('GPU'))
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Enabled memory growth for', gpu)
        except Exception as exc:
            print('Could not set memory growth for', gpu, repr(exc))

    if not tf.config.list_physical_devices('GPU'):
        print('\nRESULT: TensorFlow is not using a GPU. Training will run on CPU.')
    else:
        print('\nRESULT: TensorFlow sees at least one GPU.')
except Exception as exc:
    print('TensorFlow import/check failed:', repr(exc))
