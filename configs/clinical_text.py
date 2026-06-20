DR_CLASS_DESCRIPTIONS = {
    0: "No apparent retinopathy: no lesions, normal retina.",
    1: "Mild nonproliferative diabetic retinopathy: microaneurysms only.",
    2: "Moderate nonproliferative diabetic retinopathy: more than microaneurysms; may include dot and blot hemorrhages, hard exudates, cotton wool spots.",
    3: "Severe nonproliferative diabetic retinopathy: extensive hemorrhages, venous beading, intraretinal microvascular abnormalities; no neovascularization.",
    4: "Proliferative diabetic retinopathy: neovascularization and or vitreous or preretinal hemorrhage.",
}

BUSI_CLASS_DESCRIPTIONS = {
    0: "Normal breast ultrasound: no visible focal lesion or suspicious abnormality; no mass, no distortion.",
    1: "Benign breast mass on ultrasound: oval shape, circumscribed margin, parallel orientation; may show posterior enhancement.",
    2: "Malignant breast mass on ultrasound: irregular shape, nonparallel orientation, spiculated or ill defined margins; may show posterior acoustic shadowing.",
}

DR_CONCEPTS = [
    "microaneurysms",
    "dot and blot hemorrhages",
    "hard exudates",
    "cotton wool spots",
    "venous beading",
    "intraretinal microvascular abnormalities",
    "neovascularization",
    "preretinal hemorrhage",
    "vitreous hemorrhage",
]

BUSI_CONCEPTS = [
    "no mass",
    "no distortion",
    "oval shape",
    "circumscribed margin",
    "parallel orientation",
    "posterior enhancement",
    "irregular shape",
    "spiculated margins",
    "nonparallel orientation",
    "posterior acoustic shadowing",
]

# Cumulative concept-to-grade masks for L_PIC.
# Each grade inherits all concepts from lower grades (clinical progression).
# Index refers to position in DR_CONCEPTS / BUSI_CONCEPTS lists above.
DR_CONCEPT_GRADE_MASK = {
    0: [],                         # No DR: no pathological findings
    1: [0],                        # Mild NPDR: microaneurysms
    2: [0, 1, 2, 3],               # Moderate NPDR: + hemorrhages, hard exudates, cotton wool spots
    3: [0, 1, 2, 3, 4, 5],         # Severe NPDR: + venous beading, IRMA
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8],  # PDR: all concepts
}

# Concepts 0-1 ("no mass", "no distortion") are absence-type for BUSI;
# the ConceptSpine treats them with sign-flipped weights (w <= 0).
BUSI_CONCEPT_GRADE_MASK = {
    0: [0, 1],              # Normal: absence concepts active
    1: [0, 1, 2, 3, 4, 5], # Benign: absence + benign morphology
    2: [6, 7, 8, 9],        # Malignant: malignant morphology only
}

BUSI_ABSENCE_CONCEPT_INDICES = [0, 1]  # "no mass", "no distortion"
