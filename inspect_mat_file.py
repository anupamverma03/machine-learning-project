import scipy.io as sio
import numpy as np

data = sio.loadmat("wider_easy_val.mat")

print("Top-level keys in .mat file:")
for k in data.keys():
    if not k.startswith("__"):
        print(" ", k, type(data[k]), "shape:", getattr(data[k], "shape", None))

# Try to print a small sample of each variable
print("\nSample contents:")
for k, v in data.items():
    if k.startswith("__"):
        continue
    print(f"\nVariable: {k}")
    print("Type:", type(v))
    try:
        if isinstance(v, np.ndarray):
            print("dtype:", v.dtype, "shape:", v.shape)
            print("sample element:", v.flatten()[0])
        else:
            print("value:", v)
    except Exception as e:
        print("Could not display:", e)
