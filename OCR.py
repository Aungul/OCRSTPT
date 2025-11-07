import argparse
from pathlib import Path
from datetime import datetime
import threading
import queue
import os
import sys

import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =========== OCR core (din scriptul tău, cu mici corecții) ===========

# === Robust deskew (OSD + Hough + clamp) ===
def _coarse_rotate_osd(gray: np.ndarray) -> np.ndarray:
    """Corecție grosieră 0/90/180/270 folosind OSD din Tesseract."""
    try:
        osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
        angle = int(osd.get("rotate", 0))  # 0, 90, 180, 270
    except Exception:
        angle = 0
    if angle:
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)  # negativ ca să corectăm
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return gray

def _deskew_small(gray: np.ndarray, max_abs_angle: float = 15.0) -> np.ndarray:
    """Deskew fin pe ±max_abs_angle folosind linii de text (Hough)."""
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, 1)

    lines = cv2.HoughLines(th, 1, np.pi/180, threshold=200)
    angle = 0.0
    if lines is not None:
        angs = []
        for rho_theta in lines[:200]:
            theta = rho_theta[0][1]
            deg = theta * 180.0 / np.pi
            if deg > 90: deg -= 180
            if deg < -90: deg += 180
            angs.append(deg)
        if angs:
            angle = float(np.median(angs))

    if abs(angle) > max_abs_angle:
        angle = 0.0

    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def robust_deskew(gray: np.ndarray, use_osd: bool = True, max_abs_angle: float = 15.0) -> np.ndarray:
    """Combinație: OSD (0/90/180/270) + deskew fin ±max_abs_angle."""
    if use_osd:
        gray = _coarse_rotate_osd(gray)
    return _deskew_small(gray, max_abs_angle=max_abs_angle)

def preprocess_for_ocr(
    bgr: np.ndarray,
    scale_factor: float = 1.5,
    adaptive_block: int = 31,
    adaptive_C: int = 10,
    use_osd: bool = True,
    max_abs_angle: float = 15.0
) -> np.ndarray:
    """Pipeline pentru documente scanate: robust deskew + normalizare + binarizare."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deskew robust (previne rotiri la 90°)
    gray = robust_deskew(gray, use_osd=use_osd, max_abs_angle=max_abs_angle)

    # Normalizează iluminarea (pe float pentru a evita banding)
    bg = cv2.medianBlur(gray, 39)  # 31–51 dacă umbre puternice
    gray_f = gray.astype(np.float32)
    bg_f = bg.astype(np.float32) + 1e-3
    norm_f = (gray_f / bg_f) * 255.0
    norm = np.clip(norm_f, 0, 255).astype(np.uint8)

    # Contrast local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7))
    enh = clahe.apply(norm)

    # Binarizare adaptivă (robustă pe scanuri/foto)
    bin_img = cv2.adaptiveThreshold(
        enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        adaptive_block, adaptive_C
    )

    # Mic close pentru litere rupte + scale up pentru Tesseract
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    m = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, 1)
    sc = cv2.resize(m, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return sc


def ocr_image(img_for_ocr: np.ndarray, lang="ron+eng", psm=4) -> tuple[str, float]:
    cfg = f"--oem 1 --psm {psm} -l {lang} -c user_defined_dpi=300 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(img_for_ocr, config=cfg)
    data = pytesseract.image_to_data(img_for_ocr, config=cfg, output_type=Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return text, mean_conf

def pil_to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def process_pdf(pdf_path: Path, outdir: Path, dpi=300, lang="ron+eng", psm=4,
                save_debug=False, start=None, end=None, poppler_path=None, log=lambda *_: None):
    outdir.mkdir(parents=True, exist_ok=True)
    dbgdir = outdir / "debug"
    if save_debug:
        dbgdir.mkdir(exist_ok=True)

    # Convertim paginile PDF la imagini (ai nevoie de Poppler; dacă e pe Windows, poți pasa poppler_path)
    pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=start, last_page=end,
                              poppler_path=poppler_path)

    combined_text = []
    page_stats = []

    for idx, page in enumerate(pages, 1 if start is None else start):
        bgr = pil_to_bgr(page)

        if save_debug:
            cv2.imwrite(str(dbgdir / f"{idx:03d}_orig.png"), bgr)

        prep = preprocess_for_ocr(bgr)
        if save_debug:
            cv2.imwrite(str(dbgdir / f"{idx:03d}_prep.png"), prep)

        text, mean_conf = ocr_image(prep, lang=lang, psm=psm)
        (outdir / "pages").mkdir(exist_ok=True)
        page_txt = outdir / "pages" / f"page_{idx:03d}.txt"
        page_txt.write_text(text, encoding="utf-8")

        combined_text.append(text)
        page_stats.append((idx, mean_conf))
        log(f"[pagina {idx}] mean_conf={mean_conf:.2f}")

    # Scriem textul combinat & statistici
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = outdir / f"pdf_ocr_{ts}.txt"
    combined_path.write_text("\n\n".join(combined_text), encoding="utf-8")

    stats_path = outdir / f"stats_{ts}.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        for i, c in page_stats:
            f.write(f"page {i:03d}: mean_conf={c:.2f}\n")

    log("✅ Gata.")
    log(f"📝 Text combinat: {combined_path}")
    log(f"📈 Statistici:    {stats_path}")
    if save_debug:
        log(f"🔍 Debug images:  {dbgdir}")

# =========== GUI (Tkinter) ===========

class OCRGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Scan OCR (Tesseract)")
        self.geometry("820x560")
        self.resizable(True, True)

        # Vars
        self.pdf_path   = tk.StringVar()
        self.out_dir    = tk.StringVar(value=str(Path("pdf_ocr_output").absolute()))
        self.dpi        = tk.IntVar(value=300)
        self.lang       = tk.StringVar(value="ron+eng")
        self.psm        = tk.StringVar(value="4")
        self.debug      = tk.BooleanVar(value=False)
        self.start_page = tk.StringVar()
        self.end_page   = tk.StringVar()
        self.poppler    = tk.StringVar()  # folderul bin poppler (opțional, Windows)
        self.tesseract  = tk.StringVar()  # calea către tesseract.exe (opțional)

        self._build_ui()

        # logger queue pentru thread
        self.log_q = queue.Queue()
        self.after(100, self._poll_log)

        # worker thread handle
        self.worker = None

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        # Row 1: PDF + Browse
        frm1 = ttk.Frame(self); frm1.pack(fill='x', **pad)
        ttk.Label(frm1, text="PDF:").pack(side='left')
        ttk.Entry(frm1, textvariable=self.pdf_path, width=70).pack(side='left', padx=6, fill='x', expand=True)
        ttk.Button(frm1, text="Alege…", command=self._choose_pdf).pack(side='left')

        # Row 2: Outdir + Browse
        frm2 = ttk.Frame(self); frm2.pack(fill='x', **pad)
        ttk.Label(frm2, text="Output:").pack(side='left')
        ttk.Entry(frm2, textvariable=self.out_dir, width=60).pack(side='left', padx=6, fill='x', expand=True)
        ttk.Button(frm2, text="Folder…", command=self._choose_outdir).pack(side='left')

        # Row 3: DPI, Lang, PSM, Debug
        frm3 = ttk.Frame(self); frm3.pack(fill='x', **pad)
        ttk.Label(frm3, text="DPI:").pack(side='left')
        ttk.Spinbox(frm3, from_=150, to=600, increment=50, textvariable=self.dpi, width=6).pack(side='left', padx=6)

        ttk.Label(frm3, text="Limbi:").pack(side='left')
        ttk.Entry(frm3, textvariable=self.lang, width=14).pack(side='left', padx=6)

        ttk.Label(frm3, text="PSM:").pack(side='left')
        psm_box = ttk.Combobox(frm3, textvariable=self.psm, values=["1","3","4","6","11","12"], width=6, state="readonly")
        psm_box.pack(side='left', padx=6)

        ttk.Checkbutton(frm3, text="Debug images", variable=self.debug).pack(side='left', padx=12)

        # Row 4: Start/End pages
        frm4 = ttk.Frame(self); frm4.pack(fill='x', **pad)
        ttk.Label(frm4, text="Start page:").pack(side='left')
        ttk.Entry(frm4, textvariable=self.start_page, width=8).pack(side='left', padx=6)
        ttk.Label(frm4, text="End page:").pack(side='left')
        ttk.Entry(frm4, textvariable=self.end_page, width=8).pack(side='left', padx=6)

        # Row 5: Optional paths (Poppler, Tesseract)
        frm5 = ttk.Labelframe(self, text="Căi opționale (Windows)")
        frm5.pack(fill='x', **pad)
        ttk.Label(frm5, text="Poppler bin:").grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(frm5, textvariable=self.poppler, width=64).grid(row=0, column=1, sticky='we', padx=6, pady=4)
        ttk.Button(frm5, text="Browse…", command=self._choose_poppler).grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(frm5, text="tesseract.exe:").grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(frm5, textvariable=self.tesseract, width=64).grid(row=1, column=1, sticky='we', padx=6, pady=4)
        ttk.Button(frm5, text="Browse…", command=self._choose_tesseract).grid(row=1, column=2, padx=6, pady=4)
        frm5.columnconfigure(1, weight=1)

        # Row 6: Buttons + progress
        frm6 = ttk.Frame(self); frm6.pack(fill='x', **pad)
        self.run_btn = ttk.Button(frm6, text="Rulează OCR", command=self._run_clicked)
        self.run_btn.pack(side='left')
        ttk.Button(frm6, text="Deschide output", command=self._open_outdir).pack(side='left', padx=8)

        self.pb = ttk.Progressbar(frm6, mode='indeterminate', length=200)
        self.pb.pack(side='right')

        # Row 7: Log
        frm7 = ttk.Frame(self); frm7.pack(fill='both', expand=True, **pad)
        ttk.Label(frm7, text="Log:").pack(anchor='w')
        self.log_txt = tk.Text(frm7, height=12, wrap='word')
        self.log_txt.pack(fill='both', expand=True)
        self.log_txt.configure(state='disabled')

        # Footer
        ttk.Label(self, text="Sfaturi: PSM 4 pentru pagini 1 coloană, 6 pentru bloc text; limbi: ron+eng; DPI 300.").pack(anchor='w', padx=10, pady=4)

    def _choose_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF files","*.pdf")])
        if f:
            self.pdf_path.set(f)

    def _choose_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.out_dir.set(d)

    def _choose_poppler(self):
        d = filedialog.askdirectory(title="Alege folderul bin Poppler (conține pdftoppm.exe)")
        if d:
            self.poppler.set(d)

    def _choose_tesseract(self):
        f = filedialog.askopenfilename(title="Alege tesseract.exe", filetypes=[("Executable","*.exe"),("All","*.*")])
        if f:
            self.tesseract.set(f)

    def _open_outdir(self):
        p = self.out_dir.get().strip()
        if not p:
            return
        path = Path(p)
        if path.exists():
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")

    def _log(self, msg):
        self.log_q.put(str(msg))

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log_txt.configure(state='normal')
                self.log_txt.insert('end', msg + "\n")
                self.log_txt.see('end')
                self.log_txt.configure(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _run_clicked(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("În curs", "OCR rulează deja…")
            return

        pdf = self.pdf_path.get().strip()
        if not pdf:
            messagebox.showwarning("Lipsește PDF", "Selectează un fișier PDF.")
            return

        outdir = self.out_dir.get().strip() or "pdf_ocr_output"
        dpi = int(self.dpi.get())
        lang = self.lang.get().strip() or "ron+eng"
        psm = int(self.psm.get())
        debug = bool(self.debug.get())
        start = self._parse_int(self.start_page.get())
        end   = self._parse_int(self.end_page.get())
        poppler_path = self.poppler.get().strip() or None
        tesseract_exe = self.tesseract.get().strip() or None

        # setăm tesseract dacă e specificat
        if tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe

        # pornește thread-ul de lucru
        self.run_btn.configure(state='disabled')
        self.pb.start(10)
        self.log_txt.configure(state='normal'); self.log_txt.delete('1.0', 'end'); self.log_txt.configure(state='disabled')
        self._log(f"→ PDF: {pdf}")
        self._log(f"→ Output: {outdir}")
        self._log(f"→ DPI={dpi}, PSM={psm}, Lang={lang}, Debug={debug}, Start={start}, End={end}")
        if poppler_path: self._log(f"→ Poppler bin: {poppler_path}")
        if tesseract_exe: self._log(f"→ tesseract.exe: {tesseract_exe}")

        def work():
            try:
                process_pdf(
                    pdf_path=Path(pdf),
                    outdir=Path(outdir),
                    dpi=dpi,
                    lang=lang,
                    psm=psm,
                    save_debug=debug,
                    start=start,
                    end=end,
                    poppler_path=poppler_path,
                    log=self._log
                )
            except Exception as e:
                self._log(f"❌ Eroare: {e}")
            finally:
                self.pb.stop()
                self.run_btn.configure(state='normal')

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    @staticmethod
    def _parse_int(s):
        s = str(s).strip()
        if not s:
            return None
        try:
            return int(s)
        except:
            return None

# =========== Entry-point (opțional: permite și CLI clasic) ===========

def main_cli():
    ap = argparse.ArgumentParser(description="Pornește GUI-ul OCR pentru PDF-uri scanate.")
    ap.add_argument("--cli", action="store_true", help="Rulează în modul CLI (fără GUI) – compatibil cu scriptul anterior.")
    ap.add_argument("--pdf")
    ap.add_argument("--outdir", default="pdf_ocr_output")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", default="ron+eng")
    ap.add_argument("--psm", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--poppler_path")
    ap.add_argument("--tesseract_exe")
    args = ap.parse_args()

    if args.cli:
        if args.tesseract_exe:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_exe
        process_pdf(
            pdf_path=Path(args.pdf),
            outdir=Path(args.outdir),
            dpi=args.dpi,
            lang=args.lang,
            psm=args.psm,
            save_debug=args.debug,
            start=args.start,
            end=args.end,
            poppler_path=args.poppler_path,
            log=print
        )
    else:
        app = OCRGui()
        app.mainloop()

if __name__ == "__main__":
    main_cli()
