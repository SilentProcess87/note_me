# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for NoteMe.
Produces:  dist/NoteMe/NoteMe.exe  (onedir, no console window)

Build with:
    pyinstaller noteme.spec
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs
import sys, os, glob

block_cipher = None

# ── Collect native-heavy packages ──────────────────────────────────────
fw_datas,   fw_bins,  fw_hidden   = collect_all('faster_whisper')
ct2_datas,  ct2_bins, ct2_hidden  = collect_all('ctranslate2')
tok_datas,  tok_bins, tok_hidden  = collect_all('tokenizers')
av_datas,   av_bins,  av_hidden   = collect_all('av')
hf_datas,   hf_bins,  hf_hidden   = collect_all('huggingface_hub')
paw_datas,  paw_bins, paw_hidden  = collect_all('pyaudiowpatch')

_site = next(
    p for p in __import__('site').getsitepackages() if 'site-packages' in p
)

# sounddevice — collect robustly (the _sounddevice.py stub may not exist)
_sd_data_dir = os.path.join(_site, '_sounddevice_data')
_sd_portaudio_pyd = glob.glob(os.path.join(_site, '_portaudiowpatch*.pyd'))
sd_datas = []
for _f in [
    os.path.join(_site, 'sounddevice.py'),
    os.path.join(_site, '_sounddevice.py'),
]:
    if os.path.isfile(_f):
        sd_datas.append((_f, '.'))
if os.path.isdir(_sd_data_dir):
    sd_datas.append((_sd_data_dir, '_sounddevice_data'))
sd_bins   = [(p, '.') for p in _sd_portaudio_pyd]
sd_hidden = ['sounddevice', '_sounddevice_data']

# ── CUDA DLLs from pip-installed nvidia-* packages ─────────────────────
# Bundle them so the exe works on any NVIDIA GPU without needing pip packages.
nvidia_dlls = []
_nvidia_base = os.path.join(_site, 'nvidia')
if os.path.isdir(_nvidia_base):
    for _pkg in os.listdir(_nvidia_base):
        for _sub in ('bin', 'lib'):
            _bin_dir = os.path.join(_nvidia_base, _pkg, _sub)
            if os.path.isdir(_bin_dir):
                for _dll in glob.glob(os.path.join(_bin_dir, '*.dll')):
                    nvidia_dlls.append((_dll, '.'))
    print(f'[spec] CUDA DLLs bundled: {len(nvidia_dlls)} from {_nvidia_base}')
else:
    print(f'[spec] WARNING: no nvidia pip packages found — GPU may not work in EXE')

all_datas = (
    [('config.yaml', '.')]          # default config shipped with the app
    + fw_datas + ct2_datas + tok_datas + av_datas + sd_datas + hf_datas + paw_datas
)

all_binaries = fw_bins + ct2_bins + tok_bins + av_bins + sd_bins + hf_bins + paw_bins + nvidia_dlls

all_hidden = (
    fw_hidden + ct2_hidden + tok_hidden + av_hidden + sd_hidden + hf_hidden + paw_hidden
    + [
        # SQLAlchemy
        'sqlalchemy.dialects.sqlite',
        'sqlalchemy.dialects.sqlite.pysqlite',
        'sqlalchemy.orm',
        'sqlalchemy.pool',
        'sqlalchemy.pool.impl',
        'sqlalchemy.event',
        'sqlalchemy.event.api',
        # Pydantic
        'pydantic',
        'pydantic_core',
        # Networking
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
        # Audio
        'pyaudiowpatch',
        'sounddevice',
        # ML runtime
        'onnxruntime',
        'onnxruntime.capi',
        # Misc
        'yaml',
        'tqdm',
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'filelock',
        'packaging',
    ]
)

# ──────────────────────────────────────────────────────────────────────
a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'IPython',
        'notebook',
        'pytest',
        'torch',
        'torchvision',
        'torchaudio',
        'pygame',
        'jedi',
        'numba',
        'llama_index',
        'llama_index.core',
        # Exclude other Qt bindings to avoid the multi-binding conflict
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtWidgets',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NoteMe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # skip UPX — can break native DLLs
    console=False,      # no console window (windowed app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NoteMe',
)
