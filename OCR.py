import argparse
from pathlib import Path
from datetime import datetime
import threading
import queue
import os
import sys
import re
import textwrap

import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ===================== Helpers: base/bin detection & health check =====================

def _frozen_base_dir(log_fn=None) -> Path:
    """Returnează folderul de bază al aplicației:
       - în one-folder/one-file: folderul exe-ului
       - în dev: folderul scriptului .py
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        if log_fn:
            log_fn(f"[i] frozen=True  exe={sys.executable}")
            log_fn(f"[i] base_dir={base}")
        return base
    base = Path(__file__).parent
    if log_fn:
        log_fn(f"[i] frozen=False script_dir={base}")
    return base

def _bin_dir(log_fn=None) -> Path:
    """În build: <exe>/bin; în dev: thirdparty/poppler/bin (fallback)."""
    base = _frozen_base_dir(log_fn)
    candidate = base / "bin"
    if candidate.exists():
        if log_fn: log_fn(f"✅ BIN folder: {candidate}")
        return candidate
    dev_poppler_bin = Path(__file__).parent / "bin"
    if log_fn: log_fn(f"⚠️ Fallback DEV poppler bin: {dev_poppler_bin}")
    return dev_poppler_bin

def _health_check(bin_path: Path, log_fn=None) -> dict:
    """Verifică rapid prezența binarelor & datelor necesare și setează tesseract_cmd."""
    def _log(msg): (log_fn or print)(msg)

    tesseract_exe = bin_path / "tesseract.exe"
    tessdata_dir  = bin_path / "tessdata"
    osd_file      = tessdata_dir / "osd.traineddata"
    eng_file      = tessdata_dir / "eng.traineddata"
    ron_file      = tessdata_dir / "ron.traineddata"
    pdftoppm_exe  = bin_path / "pdftoppm.exe"
    pdftocairo_exe= bin_path / "pdftocairo.exe"

    checks = {
        "tesseract_exe": tesseract_exe.exists(),
        "tessdata_dir":  tessdata_dir.exists(),
        "osd_trained":   osd_file.exists(),
        "eng_trained":   eng_file.exists(),
        "ron_trained":   ron_file.exists(),
        "pdftoppm":      pdftoppm_exe.exists(),
        "pdftocairo":    pdftocairo_exe.exists(),
    }

    _log(f"[check] tesseract.exe: {checks['tesseract_exe']}  -> {tesseract_exe}")
    _log(f"[check] tessdata/:     {checks['tessdata_dir']}  -> {tessdata_dir}")
    _log(f"[check]   osd.traineddata: {checks['osd_trained']}")
    _log(f"[check]   eng.traineddata: {checks['eng_trained']}")
    _log(f"[check]   ron.traineddata: {checks['ron_trained']}")
    _log(f"[check] pdftoppm.exe:  {checks['pdftoppm']}  -> {pdftoppm_exe}")
    _log(f"[check] pdftocairo.exe:{checks['pdftocairo']} -> {pdftocairo_exe}")

    if checks["tesseract_exe"]:
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)
        _log(f"[set] pytesseract.tesseract_cmd = {tesseract_exe}")

    if not checks["osd_trained"]:
        _log("⚠️ Lipsă osd.traineddata → OSD (rotire grosieră) poate eșua.")
    if not checks["pdftoppm"]:
        _log("⚠️ Lipsă pdftoppm.exe → conversia PDF→imagine (pdf2image) va eșua pe Windows.")

    return checks

# ===================== OCR core =====================

# --- Robust deskew (OSD + Hough + clamp) ---
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
    max_abs_angle: float =1.0
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

# ---------- Paragraf filtering (min words + mean conf) ----------
def _format_paragraph(text: str, width: int = 100) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([(\[{])\s+", r"\1", t)
    t = re.sub(r"\s+([)\]}])", r"\1", t)
    return textwrap.fill(t, width=width)

def _group_paragraphs_min_words_and_conf(data, min_words_per_par=8, min_par_mean_conf=65):
    """
    Grupează cuvintele după (block_num, par_num), calculează mean_conf pe paragraf
    și păstrează doar paragrafele cu >= min_words_per_par și mean_conf >= min_par_mean_conf.
    Returnează listă de (text_paragraf, mean_conf_paragraf) în ordinea de citire.
    """
    n = len(data["text"])
    def to_int_list(lst):
        out = []
        for v in lst:
            try: out.append(int(v))
            except: out.append(-1)
        return out

    level  = to_int_list(data.get("level", [""]*n))
    block  = to_int_list(data.get("block_num", [""]*n))
    parnum = to_int_list(data.get("par_num", [""]*n))
    left   = to_int_list(data.get("left", [""]*n))
    top    = to_int_list(data.get("top", [""]*n))
    words  = data.get("text", [""]*n)

    confs = []
    for v in data.get("conf", [""]*n):
        try: confs.append(int(v))
        except: confs.append(-1)

    buckets = {}
    for i in range(n):
        if level[i] != 5:  # word level
            continue
        if confs[i] < 0:
            continue
        tok = (words[i] or "").strip()
        if not tok:
            continue
        key = (block[i], parnum[i])
        if key not in buckets:
            buckets[key] = {"tokens": [], "confs": [], "tops": [], "lefts": []}
        buckets[key]["tokens"].append(tok)
        buckets[key]["confs"].append(confs[i])
        buckets[key]["tops"].append(top[i])
        buckets[key]["lefts"].append(left[i])

    paras = []
    for key, B in buckets.items():
        if len(B["tokens"]) < min_words_per_par:
            continue
        mc = float(np.mean(B["confs"])) if B["confs"] else 0.0
        if mc < min_par_mean_conf:
            continue
        txt = " ".join(B["tokens"])
        t_med = int(np.median(B["tops"])) if B["tops"] else 0
        l_med = int(np.median(B["lefts"])) if B["lefts"] else 0
        paras.append((t_med, l_med, txt, mc))

    paras.sort(key=lambda x: (x[0], x[1]))
    return [(p[2], p[3]) for p in paras]

def ocr_paragraphs_filtered(img_for_ocr: np.ndarray, lang="ron+eng", psm=4,
                            min_words_per_par=8, min_par_mean_conf=65, wrap_width=100):
    """
    OCR -> paragrafe filtrate după #cuvinte și mean_conf pe paragraf -> formatare frumoasă.
    Întoarce (text_concat_curat, mean_conf_global, lista_paragrafe_curate).
    """
    cfg = f"--oem 1 --psm {psm} -l {lang} -c user_defined_dpi=300 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img_for_ocr, config=cfg, output_type=Output.DICT)

    confs_all = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
    mean_conf_global = float(np.mean(confs_all)) if confs_all else 0.0

    raw_paras = _group_paragraphs_min_words_and_conf(
        data,
        min_words_per_par=min_words_per_par,
        min_par_mean_conf=min_par_mean_conf
    )

    formatted = [_format_paragraph(txt, width=wrap_width) for (txt, _mc) in raw_paras]
    text_concat = "\n\n".join(formatted)
    return text_concat, mean_conf_global, raw_paras

def pil_to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def process_pdf(pdf_path: Path, outdir: Path, dpi=300, lang="ron+eng", psm=4,
                save_debug=False, start=None, end=None, log=lambda *_: None,
                use_paragraph_filter=True, min_words_per_par=8, min_par_mean_conf=65, wrap_width=100,
                poppler_path: str | None = None):
    outdir.mkdir(parents=True, exist_ok=True)
    dbgdir = outdir / "debug"
    if save_debug:
        dbgdir.mkdir(exist_ok=True)

    # Convertim paginile PDF la imagini; pe Windows e necesar poppler_path
    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=start,
        last_page=end,
        poppler_path=poppler_path
    )

    combined_text_raw = []
    combined_text_clean = []
    page_stats = []

    for idx, page in enumerate(pages, 1 if start is None else start):
        bgr = pil_to_bgr(page)

        if save_debug:
            cv2.imwrite(str(dbgdir / f"{idx:03d}_orig.png"), bgr)

        prep = preprocess_for_ocr(bgr)
        if save_debug:
            cv2.imwrite(str(dbgdir / f"{idx:03d}_prep.png"), prep)

        # RAW (brut)
        raw_text, raw_mean_conf = ocr_image(prep, lang=lang, psm=psm)
        if save_debug:
            (outdir / "pagini_debug").mkdir(exist_ok=True)
        (outdir / "pagini_individuale").mkdir(exist_ok=True)
        if save_debug:
            page_txt_raw = outdir / "pagini_debug" / f"{pdf_path.name}_pagina_{idx:03d}_debug.txt"
            page_txt_raw.write_text(raw_text, encoding="utf-8")
            combined_text_raw.append(raw_text)

        # CLEAN (paragrafe filtrate + formatate)
        if use_paragraph_filter:
            clean_text, mean_conf_global, _raw_paras = ocr_paragraphs_filtered(
                prep, lang=lang, psm=psm,
                min_words_per_par=min_words_per_par,
                min_par_mean_conf=min_par_mean_conf,
                wrap_width=wrap_width
            )
            page_txt_clean = outdir / "pagini_individuale" / f"{pdf_path.name}_page_{idx:03d}.txt"
            page_txt_clean.write_text(clean_text, encoding="utf-8")
            combined_text_clean.append(clean_text)
            page_stats.append((idx, mean_conf_global))
            log(f"[pagina {idx}] conf_global={mean_conf_global:.2f} (raw_mean={raw_mean_conf:.2f})")
        else:
            page_stats.append((idx, raw_mean_conf))
            log(f"[pagina {idx}] mean_conf={raw_mean_conf:.2f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Scriem combinatul brut
    if save_debug:
        combined_path_raw = outdir / f"{pdf_path.name}_{ts}_debug.txt"
        combined_path_raw.write_text("\n\n".join(combined_text_raw), encoding="utf-8")
        log(f"📝 Text combinat (raw):   {combined_path_raw}")

    # Scriem combinatul curat (dacă s-a folosit filtrarea)
    if use_paragraph_filter:
        combined_path_clean = outdir / f"{pdf_path.name}_{ts}.txt"
        combined_path_clean.write_text("\n\n".join(combined_text_clean), encoding="utf-8")
        log(f"🧹 Text combinat (clean): {combined_path_clean}")

    # Statistici (pe pagina, din global/ raw)
    if save_debug:
        stats_path = outdir / f"stats_{ts}.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            for i, c in page_stats:
                f.write(f"page {i:03d}: mean_conf={c:.2f}\n")
        log(f"📈 Statistici: {stats_path}")

    if save_debug:
        log(f"🔍 Debug images: {dbgdir}")

    log("✅ Gata.")

# ===================== GUI (Tkinter) =====================

class OCRGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Scan OCR (Tesseract)")
        self.geometry("900x680")
        self.resizable(True, True)

        # Vars
        self.pdf_path   = tk.StringVar()
        self.out_dir    = tk.StringVar(value=str(Path("pdf_ocr_output").absolute()))
        self.dpi        = tk.IntVar(value=300)
        self.lang       = tk.StringVar(value="ron")
        self.psm        = tk.StringVar(value="4")
        self.debug      = tk.BooleanVar(value=False)
        self.start_page = tk.StringVar()
        self.end_page   = tk.StringVar()
        self.doc_path   = tk.StringVar(value="Documentatie.pdf")

        # Paragraph filtering (debuggable)
        self.use_par_filter     = tk.BooleanVar(value=True)
        self.min_words_per_par  = tk.IntVar(value=8)
        self.min_par_mean_conf  = tk.IntVar(value=65)
        self.wrap_width         = tk.IntVar(value=100)

        self._build_ui()

        self.log_q = queue.Queue()
        self.after(100, self._poll_log)
        self.worker = None

        # Startup info
        _ = _frozen_base_dir(self._log)
        _ = _bin_dir(self._log)

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

        ttk.Checkbutton(frm3, text="Debug", variable=self.debug).pack(side='left', padx=12)

        # Row 4: Start/End pages
        frm4 = ttk.Frame(self); frm4.pack(fill='x', **pad)
        ttk.Label(frm4, text="Pagina start:").pack(side='left')
        ttk.Entry(frm4, textvariable=self.start_page, width=8).pack(side='left', padx=6)
        ttk.Label(frm4, text="Pagina sfarsit:").pack(side='left')
        ttk.Entry(frm4, textvariable=self.end_page, width=8).pack(side='left', padx=6)

        # Row 5: Paragraph filtering (debug)
        frm5 = ttk.Labelframe(self, text="Paragraph filtering")
        frm5.pack(fill='x', **pad)
        ttk.Checkbutton(frm5, text="Use paragraph filtering", variable=self.use_par_filter).grid(row=0, column=0, sticky='w', padx=6, pady=4, columnspan=2)

        ttk.Label(frm5, text="Cuvinte min. per paragraf:").grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Spinbox(frm5, from_=1, to=50, textvariable=self.min_words_per_par, width=6).grid(row=1, column=1, sticky='w', padx=6, pady=4)

        ttk.Label(frm5, text="Min mean conf (paragraph):").grid(row=1, column=2, sticky='w', padx=6, pady=4)
        ttk.Spinbox(frm5, from_=0, to=100, textvariable=self.min_par_mean_conf, width=6).grid(row=1, column=3, sticky='w', padx=6, pady=4)

        ttk.Label(frm5, text="Wrap width (chars):").grid(row=1, column=4, sticky='w', padx=6, pady=4)
        ttk.Spinbox(frm5, from_=40, to=160, textvariable=self.wrap_width, width=6).grid(row=1, column=5, sticky='w', padx=6, pady=4)

        for c in range(6):
            frm5.columnconfigure(c, weight=1)

        # Row 6: Buttons + progress
        frm6 = ttk.Frame(self); frm6.pack(fill='x', **pad)
        self.run_btn = ttk.Button(frm6, text="Rulează OCR", command=self._run_clicked)
        self.run_btn.pack(side='left')
        ttk.Button(frm6, text="Deschide output", command=self._open_outdir).pack(side='left', padx=8)
        ttk.Button(frm6, text="Deschide doc", command=self._open_doc).pack(side="left", padx=8)

        self.pb = ttk.Progressbar(frm6, mode='indeterminate', length=220)
        self.pb.pack(side='right')

        # Row 7: Log
        frm7 = ttk.Frame(self); frm7.pack(fill='both', expand=True, **pad)
        ttk.Label(frm7, text="Log:").pack(anchor='w')
        self.log_txt = tk.Text(frm7, height=14, wrap='word')
        self.log_txt.pack(fill='both', expand=True)
        self.log_txt.configure(state='disabled')

        # Footer
        ttk.Label(self, text="PSM 4 = 1 coloană; PSM 6 = bloc text; limbi ex. ron+eng; DPI 300.").pack(anchor='w', padx=10, pady=4)

    def _choose_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF files","*.pdf")])
        if f:
            self.pdf_path.set(f)

    def _choose_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.out_dir.set(d)

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

    def _open_doc(self):
    # baza e folderul binarului
        base = _bin_dir(self._log)
        if not base:
            return
        
        doc_file = self.doc_path.get().strip()

        doc_path = base / doc_file  # construim calea completă
        if doc_path.exists():
            if sys.platform.startswith("win"):
                os.startfile(str(doc_path))
            elif sys.platform == "darwin":
                os.system(f"open '{doc_path}'")
            else:
                os.system(f"xdg-open '{doc_path}'")
        else:
            self._log(f"⚠️ Documentația nu a fost găsită: {doc_path}")

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

        use_par_filter = bool(self.use_par_filter.get())
        min_words = int(self.min_words_per_par.get())
        min_conf  = int(self.min_par_mean_conf.get())
        wrap_w    = int(self.wrap_width.get())

        # determină BIN PATH + health check
        bin_path = _bin_dir(self._log)
        self._log(f"[info] BIN PATH → {bin_path}")
        _ = _health_check(bin_path, self._log)

        # pornește thread-ul de lucru
        self.run_btn.configure(state='disabled')
        self.pb.start(10)
        self.log_txt.configure(state='normal'); self.log_txt.delete('1.0', 'end'); self.log_txt.configure(state='disabled')
        self._log(f"→ PDF: {pdf}")
        self._log(f"→ Output: {outdir}")
        self._log(f"→ DPI={dpi}, PSM={psm}, Lang={lang}, Debug={debug}, Start={start}, End={end}")
        self._log(f"→ Paragraph filter: use={use_par_filter}, min_words={min_words}, min_conf={min_conf}, wrap={wrap_w}")

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
                    log=self._log,
                    use_paragraph_filter=use_par_filter,
                    min_words_per_par=min_words,
                    min_par_mean_conf=min_conf,
                    wrap_width=wrap_w,
                    poppler_path=str(bin_path)  # important pe Windows
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

# ===================== Entry-point =====================

def main_cli():
    ap = argparse.ArgumentParser(description="Pornește GUI-ul OCR pentru PDF-uri scanate.")
    ap.add_argument("--cli", action="store_true", help="Rulează în modul CLI (fără GUI).")
    ap.add_argument("--pdf")
    ap.add_argument("--outdir", default="pdf_ocr_output")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", default="ron+eng")
    ap.add_argument("--psm", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--use_par_filter", action="store_true")
    ap.add_argument("--min_words_per_par", type=int, default=8)
    ap.add_argument("--min_par_mean_conf", type=int, default=65)
    ap.add_argument("--wrap_width", type=int, default=100)
    args = ap.parse_args()

    if args.cli:
        bin_path = _bin_dir(print)
        _ = _health_check(bin_path, print)
        process_pdf(
            pdf_path=Path(args.pdf),
            outdir=Path(args.outdir),
            dpi=args.dpi,
            lang=args.lang,
            psm=args.psm,
            save_debug=args.debug,
            start=args.start,
            end=args.end,
            log=print,
            use_paragraph_filter=args.use_par_filter,
            min_words_per_par=args.min_words_per_par,
            min_par_mean_conf=args.min_par_mean_conf,
            wrap_width=args.wrap_width,
            poppler_path=str(bin_path)
        )
    else:
        app = OCRGui()
        app.mainloop()

if __name__ == "__main__":
    main_cli()
