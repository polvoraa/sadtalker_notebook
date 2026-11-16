import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "SadTalker_Colab.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# SadTalker oficial no Google Colab

Notebook para instalar e executar o [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) com a interface Gradio e uma GPU T4.

## Como usar

1. No Colab, selecione **Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU**.
2. Execute as células em ordem, do início ao fim.
3. Na última célula de inicialização, abra o link `.gradio.live`.
4. Envie a imagem e o áudio na interface e clique em **Generate**. O padrão de enquadramento é `full` para preservar a imagem inteira.
5. Se desejar, execute a célula de monitoramento depois de iniciar uma geração.

O link Gradio é público enquanto a sessão estiver ativa. Não envie conteúdo privado ou confidencial.
"""
    ),
    markdown("## 1. Conferir GPU e armazenamento\n"),
    code(
        """import shutil
import subprocess

gpu = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
print(gpu.stdout or gpu.stderr)
if gpu.returncode != 0:
    raise RuntimeError("GPU NVIDIA não detectada. Ative a GPU T4 nas configurações de runtime do Colab.")

disk = shutil.disk_usage("/content")
print(f"Espaço livre em /content: {disk.free / (1024**3):.1f} GiB")
if disk.free < 12 * 1024**3:
    raise RuntimeError("Há menos de 12 GiB livres; libere espaço antes de baixar dependências e modelos.")
"""
    ),
    markdown("## 2. Instalar `uv`, Python 3.10 e criar o ambiente isolado\n"),
    code(
        """import os
import shutil
import subprocess
import sys
from pathlib import Path

VENV = Path("/content/sadtalker-venv")
PY310 = VENV / "bin" / "python"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("O comando uv não foi encontrado após a instalação.")

subprocess.run([UV, "python", "install", "3.10"], check=True)
if not PY310.exists():
    subprocess.run([UV, "venv", "--python", "3.10", str(VENV)], check=True)

version = subprocess.check_output(
    [str(PY310), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
    text=True,
).strip()
if version != "3.10":
    raise RuntimeError(f"O ambiente isolado usa Python {version}, não Python 3.10.")
print("Ambiente isolado pronto:", PY310)
"""
    ),
    markdown("## 3. Obter o repositório oficial\n"),
    code(
        """import subprocess
from pathlib import Path

REPO = Path("/content/SadTalker")
if not (REPO / ".git").exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/OpenTalker/SadTalker.git", str(REPO)],
        check=True,
    )
else:
    print("Repositório oficial já existe; mantendo o checkout atual.")

print("SadTalker:", REPO)
"""
    ),
    markdown("## 4. Instalar PyTorch com CUDA 12.1 para a T4\n"),
    code(
        """import shutil
import subprocess
from pathlib import Path

PY310 = Path("/content/sadtalker-venv/bin/python")
UV = shutil.which("uv")
if UV is None or not PY310.exists():
    raise RuntimeError("Execute as células de instalação do Python 3.10 primeiro.")

subprocess.run(
    [
        UV, "pip", "install", "--python", str(PY310),
        "torch==2.3.1", "torchvision==0.18.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ],
    check=True,
)
"""
    ),
    markdown(
        """## 5. Instalar dependências compatíveis

O pacote `av` foi intencionalmente excluído: sua compilação no Colab requer bibliotecas de sistema que não são necessárias para o fluxo de imagem + áudio do SadTalker. O `basicsr` é instalado separadamente, sem build isolation, depois do PyTorch e do NumPy.
"""
    ),
    code(
        """import shutil
import subprocess
from pathlib import Path

PY310 = Path("/content/sadtalker-venv/bin/python")
UV = shutil.which("uv")

def install(*packages, extra_args=()):
    command = [UV, "pip", "install", "--python", str(PY310), *extra_args, *packages]
    subprocess.run(command, check=True)

# Pin versions that avoid the Python 3.12/3.13 and NumPy 2.x incompatibilities.
install(
    "setuptools==80.9.0",
    "numpy==1.23.5",
    "Cython==0.29.36",
    "Pillow==9.5.0",
    "scipy==1.10.1",
    "opencv-python==4.8.1.78",
    "pydantic==1.10.15",
)

# Install the remaining runtime dependencies without the optional PyAV build.
install(
    "face-alignment==1.3.5",
    "imageio==2.19.3",
    "imageio-ffmpeg==0.4.7",
    "librosa==0.9.2",
    "numba==0.57.1",
    "resampy==0.3.1",
    "pydub==0.25.1",
    "kornia==0.6.8",
    "tqdm",
    "yacs==0.1.8",
    "PyYAML",
    "joblib==1.1.0",
    "scikit-image==0.19.3",
    "safetensors==0.4.3",
    "gradio==3.50.2",
    "huggingface_hub==0.25.2",
    "lmdb",
    "addict",
    "yapf",
    "requests",
    "future",
)

# BasicSR 1.4.2 has legacy setup requirements; use the already installed
# NumPy/PyTorch instead of an isolated build environment.
install(
    "basicsr==1.4.2",
    extra_args=("--no-build-isolation", "--no-deps"),
)
install("facexlib==0.3.0", extra_args=("--no-deps",))
install("gfpgan==1.3.8", extra_args=("--no-deps",))

print("Instalação das dependências concluída.")
"""
    ),
    markdown("## 6. Instalar FFmpeg\n"),
    code(
        """import subprocess

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
subprocess.run(["ffmpeg", "-version"], check=True)
"""
    ),
    markdown(
        """## 7. Fixar versões web compatíveis

O Gradio 3.50.2 usa APIs antigas do FastAPI/Starlette e espera compatibilidade com Pydantic v1. Fixamos as versões para evitar HTTP 500 ao abrir a página (`TypeError: unhashable type: 'dict'` ou `FieldInfo has no attribute in_`). Também mantemos uma versão do HTTPX compatível com o cliente desta geração do Gradio.
"""
    ),
    code(
        """import shutil
import subprocess
from pathlib import Path

PY310 = Path("/content/sadtalker-venv/bin/python")
UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("uv não está instalado; execute primeiro as células de setup.")

subprocess.run(
    [
        UV, "pip", "install", "--python", str(PY310),
        "fastapi==0.103.2",
        "starlette==0.27.0",
        "httpx==0.27.2",
        "pydantic==1.10.15",
    ],
    check=True,
)

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["MPLCONFIGDIR"] = "/content/matplotlib-config"
subprocess.run(
    [
        str(PY310), "-c",
        "import fastapi, starlette, httpx, gradio; "
        "print('FastAPI:', fastapi.__version__); "
        "print('Starlette:', starlette.__version__); "
        "print('HTTPX:', httpx.__version__); "
        "print('Pydantic:', __import__('pydantic').__version__); "
        "print('Gradio:', gradio.__version__)",
    ],
    env=env,
    check=True,
)
"""
    ),
    markdown(
        """## 8. Aplicar patches e testar os imports

Esta célula corrige o import do `basicsr`, desativa o TTS opcional, configura o Matplotlib antes de importar Gradio e deixa `full`, `Still Mode` e batch size 1 como padrão. Ela também testa os imports usando o Python 3.10 real do ambiente isolado.
"""
    ),
    code(
        """import os
import subprocess
from pathlib import Path

REPO = Path("/content/SadTalker")
PY310 = Path("/content/sadtalker-venv/bin/python")
APP = REPO / "app_sadtalker.py"

patch_script = r'''
import ast
import importlib.util
from pathlib import Path

repo = Path("/content/SadTalker")
app_path = repo / "app_sadtalker.py"

# Patch BasicSR's removed torchvision import.
spec = importlib.util.find_spec("basicsr")
if spec is None or spec.origin is None:
    raise RuntimeError("basicsr não foi instalado no Python 3.10.")
degradations = Path(spec.origin).parent / "data" / "degradations.py"
text = degradations.read_text(encoding="utf-8")
text = text.replace(
    "torchvision.transforms.functional_tensor",
    "torchvision.transforms.functional",
)
degradations.write_text(text, encoding="utf-8")
if "torchvision.transforms.functional_tensor" in degradations.read_text(encoding="utf-8"):
    raise RuntimeError("O patch do basicsr não foi aplicado.")

source = app_path.read_text(encoding="utf-8")

# Set MPLBACKEND before importing Gradio/Matplotlib.
if "import gradio as gr" not in source:
    raise RuntimeError("Não encontrei o import esperado de Gradio no app oficial.")
if 'os.environ["MPLBACKEND"] = "Agg"' not in source:
    if "import os, sys\\n" not in source:
        raise RuntimeError("Não encontrei o cabeçalho esperado para configurar MPLBACKEND.")
    source = source.replace(
        "import os, sys\\n",
        'import os, sys\\nos.environ["MPLBACKEND"] = "Agg"\\n',
        1,
    )

# Audio upload is used instead of the optional Coqui TTS feature.
tts_import = "from src.utils.text2speech import TTSTalker"
if tts_import in source:
    guard = "if sys.platform != 'win32' and not in_webui:"
    if guard in source:
        source = source.replace(guard, "if False:", 1)
    elif "if False:" not in source:
        raise RuntimeError("O bloco opcional de TTS mudou; revise o app_sadtalker.py.")

# Prefer the full original image and a T4-friendly default batch size.
replacements = [
    (
        "preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop'",
        "preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='full'",
    ),
    (
        'is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")',
        'is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)", value=True)',
    ),
    (
        'batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)',
        'batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)',
    ),
    (
        "demo.launch()",
        'demo.launch(share=True, server_name="0.0.0.0", show_error=True)',
    ),
]
for old, new in replacements:
    if new not in source:
        if old not in source:
            raise RuntimeError(f"Não encontrei o trecho esperado no app_sadtalker.py: {old}")
        source = source.replace(old, new, 1)

ast.parse(source)
app_path.write_text(source, encoding="utf-8")
print("Patches aplicados.")
'''

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["MPLCONFIGDIR"] = "/content/matplotlib-config"
test_script = patch_script + r'''
import basicsr.data.degradations
import gradio
import src.gradio_demo
import app_sadtalker
print("Imports do SadTalker, BasicSR e Gradio: OK")
print("Gradio:", gradio.__version__)
'''

result = subprocess.run(
    [str(PY310), "-c", test_script],
    cwd=REPO,
    env=env,
    text=True,
    capture_output=True,
)
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise RuntimeError("Os patches foram aplicados, mas o teste de import falhou.")
"""
    ),
    markdown(
        """## 9. Baixar e validar os modelos oficiais

O download pode levar alguns minutos. A célula torna idempotente a criação da pasta `checkpoints` antes de executar o script oficial e valida todos os arquivos usados pela interface.
"""
    ),
    code(
        """import subprocess
from pathlib import Path

REPO = Path("/content/SadTalker")
downloader = REPO / "scripts" / "download_models.sh"
if not downloader.is_file():
    raise FileNotFoundError(f"Script oficial não encontrado: {downloader}")

script = downloader.read_text(encoding="utf-8")
script = script.replace("mkdir ./checkpoints", "mkdir -p ./checkpoints", 1)
if not script.startswith("#!/"):
    script = "set -e\\n" + script
downloader.write_text(script, encoding="utf-8")

subprocess.run(["bash", str(downloader)], cwd=REPO, check=True)

required = [
    "checkpoints/mapping_00109-model.pth.tar",
    "checkpoints/mapping_00229-model.pth.tar",
    "checkpoints/SadTalker_V0.0.2_256.safetensors",
    "checkpoints/SadTalker_V0.0.2_512.safetensors",
    "gfpgan/weights/alignment_WFLW_4HG.pth",
    "gfpgan/weights/detection_Resnet50_Final.pth",
    "gfpgan/weights/GFPGANv1.4.pth",
    "gfpgan/weights/parsing_parsenet.pth",
]
missing = [
    name for name in required
    if not (REPO / name).is_file() or (REPO / name).stat().st_size < 100_000
]
if missing:
    raise RuntimeError("Modelos ausentes ou incompletos:\\n" + "\\n".join(missing))
print(f"{len(required)} arquivos de modelo validados.")
"""
    ),
    markdown(
        """## 10. Iniciar a interface Gradio

A aplicação fica em segundo plano depois que o link aparece. Mantenha esta sessão do Colab ativa para usar o link. Esta célula reinicia somente um app anterior deste repositório, para aplicar as versões web corrigidas.
"""
    ),
    code(
        """import os
import re
import signal
import subprocess
import time
from pathlib import Path
from IPython.display import HTML, Markdown, display
import requests

REPO = Path("/content/SadTalker")
PY310 = Path("/content/sadtalker-venv/bin/python")
LOG = Path("/content/sadtalker-gradio.log")
APP_PROCESS = globals().get("APP_PROCESS")

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["MPLCONFIGDIR"] = "/content/matplotlib-config"
env["PYTHONUNBUFFERED"] = "1"
env["GRADIO_ANALYTICS_ENABLED"] = "False"

# Stop only a previous SadTalker app process from this repository. This frees
# the Gradio port after updating the pinned web dependencies.
old_pids = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\\0")
        cwd = os.readlink(proc / "cwd")
        if (
            len(args) > 1
            and args[0] == str(PY310)
            and args[1] == "app_sadtalker.py"
            and cwd == str(REPO)
        ):
            pid = int(proc.name)
            os.kill(pid, signal.SIGTERM)
            old_pids.append(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
def live_process_ids(pids):
    alive = []
    for pid in pids:
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            stat = stat_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        # A terminated process can remain briefly as a zombie in /proc.
        if stat.rsplit(")", 1)[-1].strip().startswith("Z"):
            continue
        alive.append(pid)
    return alive

for _ in range(20):
    alive = live_process_ids(old_pids)
    if not alive:
        break
    time.sleep(0.5)
if alive:
    # These PIDs were matched above to the exact SadTalker command and repo cwd.
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _ in range(20):
        alive = live_process_ids(old_pids)
        if not alive:
            break
        time.sleep(0.5)
if alive:
    raise RuntimeError(f"Não consegui encerrar o servidor SadTalker antigo: {alive}")

APP_PROCESS = None
log_file = LOG.open("w", encoding="utf-8")
APP_PROCESS = subprocess.Popen(
    [str(PY310), "app_sadtalker.py"],
    cwd=REPO,
    env=env,
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

url_pattern = re.compile(r"https://[A-Za-z0-9-]+\\.gradio\\.live")
deadline = time.time() + 300
public_url = None

while time.time() < deadline:
    if APP_PROCESS.poll() is not None:
        break
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    match = url_pattern.search(log)
    if match:
        public_url = match.group(0)
        break
    time.sleep(2)

if public_url:
    local_port_match = re.search(r"localhost:(\\d+)|127\\.0\\.0\\.1:(\\d+)|0\\.0\\.0\\.0:(\\d+)", log)
    local_port = next((int(group) for group in local_port_match.groups() if group), 7860) if local_port_match else 7860
    local_response = None
    for _ in range(30):
        try:
            local_response = requests.get(f"http://127.0.0.1:{local_port}/", timeout=5)
            if local_response.status_code < 500:
                break
        except requests.RequestException:
            pass
        time.sleep(2)

    if local_response is None or local_response.status_code >= 500:
        details = LOG.read_text(encoding="utf-8", errors="replace")[-15000:] if LOG.exists() else "(log vazio)"
        print(details)
        raise RuntimeError(
            f"O link público foi criado, mas a página local retornou "
            f"{getattr(local_response, 'status_code', 'sem resposta')}."
        )

    display(HTML(f"<b>Teste local: HTTP {local_response.status_code}</b>"))
    display(Markdown(f"## SadTalker pronto\\n\\n[Abrir interface]({public_url})"))
else:
    details = LOG.read_text(encoding="utf-8", errors="replace")[-15000:] if LOG.exists() else "(log vazio)"
    print(details)
    raise RuntimeError("O servidor encerrou ou não gerou o link .gradio.live.")
"""
    ),
    markdown(
        """## 11. Monitorar uma geração

Clique em **Generate** na interface e depois execute esta célula. O SadTalker não informa a porcentagem real total; a barra é uma estimativa. A célula procura o trabalho mais recente, prefere o resultado `_full.mp4`, ignora arquivos temporários e só apresenta o vídeo quando o tamanho estiver estável. Interromper esta célula interrompe apenas o monitor, não o app.
"""
    ),
    code(
        """import html
import re
import time
from pathlib import Path
from IPython.display import FileLink, HTML, display

RESULTS = Path("/content/SadTalker/results")
MAX_MONITOR_SECONDS = 30 * 60
ESTIMATED_ANIMATION_SECONDS = 240
started = time.time()
animation_started = None
last_percent = 0
last_size = None
stable_checks = 0
active_job = None
log_path = Path("/content/sadtalker-gradio.log")
display_id = display(HTML("Aguardando nova solicitação..."), display_id=True)

def job_folders():
    if not RESULTS.exists():
        return []
    return [
        folder for folder in RESULTS.rglob("input")
        if folder.is_dir()
    ]

while time.time() - started < MAX_MONITOR_SECONDS:
    candidates = job_folders()
    newest_input = max(candidates, key=lambda p: p.stat().st_mtime, default=None)
    percent = 0
    status = "Aguardando a solicitação na interface..."
    video = None

    if newest_input is not None:
        active_job = newest_input.parent
        all_videos = list(active_job.rglob("*.mp4"))
        final_videos = [
            p for p in all_videos
            if not p.name.startswith("temp_") and not p.name.startswith("temp")
        ]
        full_videos = [p for p in final_videos if p.stem.endswith("_full")]
        if full_videos:
            video = max(full_videos, key=lambda p: p.stat().st_mtime)
        elif final_videos:
            video = max(final_videos, key=lambda p: p.stat().st_mtime)

        if video is not None:
            size = video.stat().st_size
            if size == last_size and size > 100_000:
                stable_checks += 1
            else:
                stable_checks = 0
            last_size = size
            percent = 95
            status = f"Finalizando {video.name} ({size / 1_000_000:.1f} MB)"
            if stable_checks >= 3:
                display_id.update(HTML(
                    "<b>Geração concluída: 100%</b>"
                    f"<div>{html.escape(video.name)}</div>"
                ))
                display(FileLink(str(video)))
                break
        else:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            normalized = log_text.replace("\\r", "\\n")
            mel_progress = None
            for line in normalized.splitlines():
                if "mel:" in line:
                    match = re.search(r"(\\d+)%.*?(\\d+)/(\\d+)", line)
                    if match:
                        mel_progress = round(100 * int(match.group(2)) / max(1, int(match.group(3))))

            if mel_progress is not None and mel_progress < 100:
                percent = 10 + round(25 * mel_progress / 100)
                status = f"Preparando áudio: {mel_progress}% desta etapa"
            else:
                if animation_started is None:
                    animation_started = time.time()
                elapsed = time.time() - animation_started
                estimate = min(1.0, elapsed / ESTIMATED_ANIMATION_SECONDS)
                percent = 35 + round(45 * estimate)
                status = "Gerando animação (porcentagem aproximada pelo tempo)"

    percent = max(last_percent, min(percent, 95))
    last_percent = percent
    bar = (
        f"<b>Progresso estimado: {percent}%</b>"
        '<div style="background:#ddd;border-radius:8px;height:18px;margin:6px 0">'
        f'<div style="background:#2e8b57;width:{percent}%;height:18px;border-radius:8px"></div></div>'
        f"<div>{html.escape(status)}</div>"
    )
    display_id.update(HTML(bar))
    time.sleep(5)
else:
    display_id.update(HTML(
        "Monitoramento encerrado após 30 minutos. "
        "A geração não foi interrompida; confira a interface ou execute o monitor novamente."
    ))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": NOTEBOOK.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(NOTEBOOK)
