# OCRDoc.spec
# rulat cu: pyinstaller OCRDoc.spec
from PyInstaller.utils.hooks import collect_submodules
a = Analysis(
    ['OCR.py'],
    pathex=[],
    binaries=[
        ('thirdparty/tesseract/tesseract.exe', 'bin'),
        ('thirdparty/poppler/bin/*.exe', 'bin'),
        # adaugă aici DLL-urile poppler necesare, ex:
        ('thirdparty/poppler/bin/*.dll', 'bin'),  # dacă vrei wildcard, trebuie expandat manual
    ],
    datas=[
        ('thirdparty/tesseract/tessdata', 'bin/tessdata'),
        ('licenses', 'licenses'),
        ('Documentatie.pdf', 'bin'),
    ],
    hiddenimports=collect_submodules('PIL'),
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, name='OCRDoc', console=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='OCRDoc')