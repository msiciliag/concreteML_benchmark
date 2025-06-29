For using the GPU instead of the CPU, change in the fhe_model.compile(..., device="cuda")
You also have to uninstall concrete-python with:
```bash
uv pip uninstall concrete-python
```
and install the GPU version with:
```bash
uv pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python
```
Then you can run the code with GPU support. Make sure your environment is set up correctly with the necessary CUDA libraries and drivers for your GPU.
If you encounter any issues, check the Zama documentation for troubleshooting GPU-related problems.

The only way to use the GPU without a problem is to do:
```bash
source .venv/bin/activate
```
and runnning with:
```bash
python main.py experiments_classification/tree/exp_rfclassifier_uci329.yaml --clear-progress uci329rfc.bin
python main.py experiments_classification/tree/exp_rfclassifier_uci17.yaml --clear-progress uci17rfc.bin
```

Then change:
```bash
fhe_model.compile(X_train) 
```
to
```bash
import concrete.compiler as cc
print("GPU enabled:", cc.check_gpu_enabled())
print("GPU available:", cc.check_gpu_available())
fhe_model.compile(X_train, device='cuda')
```


