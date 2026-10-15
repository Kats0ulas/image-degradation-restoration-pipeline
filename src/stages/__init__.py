"""
Τα 8 στάδια της αλυσίδας, ένα module ανά στάδιο. Κάθε module εκθέτει μια
run() που γράφει τα αποτελέσματά του στο results/stageN_.../.
"""
from . import (stage1_reference, stage2_degradation, stage3_fft, stage4_spatial,
               stage5_frequency, stage6_enhancement, stage7_edges, stage8_summary)

STAGES = {
    1: stage1_reference.run,
    2: stage2_degradation.run,
    3: stage3_fft.run,
    4: stage4_spatial.run,
    5: stage5_frequency.run,
    6: stage6_enhancement.run,
    7: stage7_edges.run,
    8: stage8_summary.run,
}
