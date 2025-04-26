# -*- mode: python ; coding: utf-8 -*-

import os
import tkinterdnd2

# --- Get EasyOCR data path reliably ---
easyocr_data_path = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
# --- Find tkinterdnd2 library path ---
tkinterdnd2_path = os.path.dirname(tkinterdnd2.__file__)
# --- ---

a = Analysis(
    ['betting_calculator.py'],
    pathex=[],
    binaries=[],
    datas=[
        (easyocr_data_path, 'easyocr/model'),
        (tkinterdnd2_path, 'tkdnd')
    ],
    hiddenimports=[
        'scipy._lib.array_api_compat.numpy.fft',
        'scipy._lib.array_api_compat.numpy',
        'scipy._lib.array_api_compat.numpy.linalg',
        'scipy.special._special_ufuncs'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BettingCalculator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BettingCalculator',
)
