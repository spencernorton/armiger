# Standalone Armiger Ultra

Use this guide to run Armiger on its own or call it from Python. If you followed the [Bitmagnet](../examples/bitmagnet/README.md) or [Rats Search](../examples/rats-search/README.md) Docker guide, you already have Armiger Ultra and can skip this page.

You need [Python 3.12](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), and internet access for the initial install and model download. No Docker, GPU, account or API key is required.

## 1. Get the files

If you already downloaded Armiger for a Docker guide, open a terminal in that repository folder and continue to step 2. Otherwise:

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger
```

## 2. Install

On **macOS or Linux**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On **Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, open Command Prompt in the same folder and use `.venv\Scripts\activate.bat` instead.

On Linux, install CPU-only PyTorch first to avoid downloading unnecessary GPU libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then, on any platform:

```bash
python -m pip install '.[ultra]'
python -m armiger download
```

Allow several minutes for dependencies. The model download itself is about 59 MB. Armiger checks the downloaded files and saves them for reuse.

## 3. Try it

```bash
python examples/classify.py
```

You should see:

```text
Content type: movie
```

The example classifies the name and file information for *Tears of Steel*, an openly licensed film. It does not download the film. Open [the example](../examples/classify.py) to see the Python call and replace its input with your own metadata.

## Next time

Open a terminal in the repository folder, activate `.venv` using the command above, and run your script. You do not need to reinstall or download the model each time. Once installed, local classification works offline.

The default model folder is `~/.cache/armiger/` under your user account. To choose a different folder or use the smaller ONNX-only model, see the [Python reference](reference.md).

If a command says Python 3.12 is missing, install that version first. If model loading fails, activate `.venv` and run `python -m armiger download` again. For more help, [report the error](https://github.com/spencernorton/armiger/issues/new/choose).
