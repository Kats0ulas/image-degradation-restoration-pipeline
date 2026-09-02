# Τελική Εργασία ΕΥ631 — Ψηφιακή Επεξεργασία Εικόνας

Αλυσίδα **Υποβάθμισης → Αποκατάστασης → Βελτίωσης** ψηφιακής εικόνας με OpenCV.

## Εγκατάσταση
```
python -m pip install -r requirements.txt
```

## Εκτέλεση
```
python main.py              # όλα τα υλοποιημένα στάδια
python main.py --stage 1    # ένα συγκεκριμένο στάδιο
```

## Δομή
```
main.py                     ενορχηστρωτής — καλεί τα στάδια με τη σειρά
requirements.txt
src/
  config.py                 όλες οι παράμετροι των πειραμάτων
  io_utils.py               φόρτωση / μετατροπή / αποθήκευση εικόνων
  viz.py                    γραφήματα (εικόνες, ιστογράμματα, πλέγματα)
  tables.py                 πίνακες αποτελεσμάτων (τερματικό / .csv / .md)
  degradation.py            Στάδιο 2 — Gaussian & Salt-and-Pepper θόρυβος
  spatial_filters.py        Στάδιο 4 — Gaussian / Median / Bilateral
  frequency_filters.py      Στάδια 3 & 5 — FFT, ιδανικό / Butterworth (n=1,2,5) / Gaussian LPF
  enhancement.py            Στάδιο 6 — Histogram Equalization, CLAHE
  edges.py                  Στάδιο 7 — Sobel, Canny
  metrics.py                PSNR / SSIM
tools/
  preview_candidates.py     βοηθητικό: προεπισκόπηση υποψήφιων εικόνων αναφοράς
images/                     οι εικόνες αναφοράς (ground truth, PNG)
results/                    όλα τα παραγόμενα αποτελέσματα ανά στάδιο
report/
  stage_notes.md            αναλυτικές σημειώσεις ανά στάδιο (τι έγινε, γιατί, τι δείχνουν τα νούμερα)
  anafora.md                πηγή της τελικής αναφοράς (Markdown)
  anafora.html              η αναφορά σε HTML (ενδιάμεσο)
  anafora.pdf               η τελική αναφορά — ΤΟ ΠΑΡΑΔΟΤΕΟ
  figs/                     υποδειγματοληπτημένα αντίγραφα των εικόνων για το PDF
tools/
  build_report.py           anafora.md -> anafora.html -> anafora.pdf
```

## Παραγωγή της αναφοράς
```
python tools/build_report.py          # HTML + PDF (headless Chrome/Edge)
python tools/build_report.py --html   # μόνο HTML
```
Δεν χρειάζεται LaTeX ή pandoc: το PDF παράγεται με `--print-to-pdf` του Chrome/Edge.
Για αρίθμηση σελίδων, άνοιξε το `report/anafora.html` στον browser και τύπωσε σε PDF
(Ctrl+P) με ενεργή την επιλογή «Κεφαλίδες και υποσέλιδα».

## Αντιστοίχιση με την εκφώνηση
| Ζητούμενο | Πού υλοποιείται | Έξοδος |
|---|---|---|
| 1. ≥2 εικόνες αναφοράς με διαφορετικά χαρακτηριστικά | `config.REFERENCE_IMAGES` | `results/stage1_reference/` |
| 2. Gaussian (≥2 σ) & S&P (≥2 d) + ιστογράμματα | `src/degradation.py` | `results/stage2_degradation/` |
| 3. Φάσμα πλάτους FFT αρχικής & υποβαθμισμένης | `src/frequency_filters.py` | `results/stage3_fft/` |
| 4. Gauss (≥2 πυρήνες), Median (≥2 παράθυρα), bilateral + PSNR/SSIM | `src/spatial_filters.py` | `results/stage4_spatial/` |
| 5. Butterworth (**≥2 τάξεις n**) & Gauss ΧΠΦ, πυρήνες + PSNR/SSIM | `src/frequency_filters.py` | `results/stage5_frequency/` |
| 6. HE + CLAHE (≥2 clipLimit) + ιστογράμματα πριν/μετά | `src/enhancement.py` | `results/stage6_enhancement/` |
| 7. Sobel (οριζόντιος/κατακόρυφος/συνδυασμένος) & Canny (≥2 κατώφλια) **στην αποκατεστημένη** | `src/edges.py` | `results/stage7_edges/` |
| 8. Συγκεντρωτικός πίνακας PSNR/SSIM ανά βήμα | `run_stage8()` | `results/stage8_summary/pipeline_summary.csv` |

## Συμβάσεις
- Όλες οι εικόνες: **grayscale, uint8, [0,255], 512×512**.
- Οι πράξεις γίνονται σε `float64` και επιστρέφουν σε `uint8` με clipping.
- Σταθερό `RANDOM_SEED = 631` → ο θόρυβος (άρα και τα PSNR/SSIM) είναι επαναλήψιμος.

## Κατάσταση σταδίων
- [x] 1. Εικόνες αναφοράς
- [x] 2. Υποβάθμιση
- [x] 3. Ανάλυση πεδίου συχνοτήτων
- [x] 4. Αποθορυβοποίηση στο χωρικό πεδίο
- [x] 5. Αποθορυβοποίηση στο πεδίο συχνοτήτων
- [x] 6. Βελτίωση εικόνας
- [x] 7. Ανίχνευση ακμών
- [x] 8. Συγκεντρωτική αξιολόγηση
- [x] Τελική αναφορά (`report/anafora.pdf`)
