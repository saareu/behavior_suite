import inspect
import platform
import sys

import torch
import sleap_nn
import sleap_io
import sleap_nn.cli

print("Python:", sys.version)
print("Platform:", platform.platform())
print("sleap-nn:", sleap_nn.__version__)
print("sleap-io:", sleap_io.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU total memory GiB:", round(props.total_memory / 1024**3, 2))
else:
    print("GPU:", None)
    print("GPU total memory GiB:", None)

print("sleap-nn package:", sleap_nn.__file__)
print("sleap-io package:", sleap_io.__file__)
print("cli source:", inspect.getsourcefile(sleap_nn.cli))
