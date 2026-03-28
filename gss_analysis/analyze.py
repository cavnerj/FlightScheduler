import pyreadr
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────
result = pyreadr.read_r('/home/user/FlightScheduler/gss_analysis/gss_all.rda')
df = result['gss_all'][['degree', 'major1', 'partyid', 'year']].copy()

print(f"Full dataset: {len(df):,} rows")

# ─────────────────────────────────────────────
# 2.  MAJOR1 code → label mapping
#     (from gssrdoc, extracted from gss_doc.rda)
# ─────────────────────────────────────────────
MAJOR1_LABELS = {
    1:  "Agriculture",
    2:  "Environment & Natural Resources",
    3:  "Architecture",
    4:  "Area / Ethnic / Civilization Studies",
    5:  "Communications",
    6:  "Communication Technologies",
    7:  "Computer & Information Sciences",
    8:  "Cosmetology / Culinary Arts",
    9:  "Education Administration & Teaching",
    10: "Engineering",
    11: "Engineering Technologies",
    12: "Linguistics & Foreign Languages",
    13: "Family & Consumer Sciences",
    14: "Law & Legal Studies",
    15: "English Language / Literature",
    16: "Liberal Arts & Humanities",
    17: "Library Science",
    18: "Biology & Life Sciences",
    19: "Mathematics & Statistics",
    20: "Military Technologies",
    21: "Interdisciplinary Studies",
    22: "Physical Fitness / Recreation",
    23: "Philosophy & Religious Studies",
    24: "Theology & Religious Vocations",
    25: "Physical Sciences",
    26: "Nuclear / Radiology / Bio Technologies",
    27: "Psychology",
    28: "Criminal Justice & Fire Protection",
    29: "Public Affairs / Policy / Social Work",
    30: "Social Sciences",
    31: "Construction Services",
    32: "Electrical & Mechanic Repair Technologies",
    33: "Precision Production / Industrial Arts",
    34: "Transportation Sciences & Technologies",
    35: "Fine Arts",
    36: "Medical & Health Sciences and Services",
    37: "Business",
    38: "History",
    39: "Other / Unspecified",
}

# ─────────────────────────────────────────────
# 3.  STEM classification
#     Judgment calls documented below
# ─────────────────────────────────────────────
#
#  STEM = codes 2, 7, 10, 11, 18, 19, 25, 26
#
#  Edge-case notes:
#   1  (Agriculture)           → non-STEM: primarily applied agri / vocational
#   2  (Environment & Nat Res) → STEM: environmental science, ecology, earth sciences
#   3  (Architecture)          → non-STEM: design-oriented, classified with professions
#   6  (Comm Technologies)     → non-STEM: media/production, not engineering
#  11  (Engineering Tech)      → STEM: applied engineering programs
#  21  (Interdisciplinary)     → non-STEM: insufficient info to classify; majority lean social/humanities
#  26  (Nuclear/Radiology/Bio) → STEM: highly technical/scientific
#  27  (Psychology)            → non-STEM: social science by convention
#  32  (Elec & Mech Repair)    → non-STEM: vocational trades, not engineering
#  36  (Medical & Health Sci)  → non-STEM: bundled with health *services*; not core STEM
#  39  (Other)                 → excluded (category undefined)
#
STEM_CODES = {2, 7, 10, 11, 18, 19, 25, 26}
NON_STEM_CODES = set(MAJOR1_LABELS.keys()) - STEM_CODES - {39}  # exclude code 39 "Other"
EXCLUDE_CODES = {39}

def classify_stem(code):
    if pd.isna(code):
        return np.nan
    code = int(code)
    if code in STEM_CODES:
        return 'STEM'
    elif code in NON_STEM_CODES:
        return 'Non-STEM'
    else:
        return np.nan  # code 39 "Other" → exclude

# ─────────────────────────────────────────────
# 4.  Recode variables
# ─────────────────────────────────────────────
# DEGREE >= 3 → bachelor's or higher
df_ba = df[df['degree'] >= 3].copy()
print(f"After DEGREE >= 3 filter: {len(df_ba):,} rows")

# PARTYID recode:  0-2=Democrat, 3=Independent, 4-6=Republican, 7=Other (exclude)
def recode_party(pid):
    if pd.isna(pid): return np.nan
    pid = int(pid)
    if pid <= 2: return 'Democrat'
    if pid == 3: return 'Independent'
    if pid <= 6: return 'Republican'
    return np.nan   # code 7 = "Other party" → exclude

df_ba['party3'] = df_ba['partyid'].map(recode_party)
df_ba['stem_cat'] = df_ba['major1'].map(classify_stem)
df_ba['major1_label'] = df_ba['major1'].map(lambda x: MAJOR1_LABELS.get(int(x), np.nan) if pd.notna(x) else np.nan)

# ─────────────────────────────────────────────
# 5.  Analysis subset: non-null for all three vars
# ─────────────────────────────────────────────
adf = df_ba.dropna(subset=['party3', 'stem_cat']).copy()
print(f"\nAnalysis subset (party3 + stem_cat non-null): {len(adf):,} rows")
print(f"  STEM:     {(adf['stem_cat']=='STEM').sum():,}")
print(f"  Non-STEM: {(adf['stem_cat']=='Non-STEM').sum():,}")
print()

# ─────────────────────────────────────────────
# 6.  Cross-tab: STEM vs non-STEM × party3
# ─────────────────────────────────────────────
PARTY_ORDER = ['Democrat', 'Independent', 'Republican']

ct = pd.crosstab(adf['stem_cat'], adf['party3'])[PARTY_ORDER]
ct['Total'] = ct.sum(axis=1)
ct_pct = ct.div(ct['Total'], axis=0) * 100

print("=" * 72)
print("TABLE 1 — STEM vs Non-STEM × 3-Category Party ID")
print("(Sample: GSS respondents with bachelor's degree or higher, MAJOR1 non-missing)")
print("=" * 72)
print(f"\n{'':20s} {'Democrat':>12s} {'Independent':>13s} {'Republican':>12s} {'Total':>8s}")
print("-" * 72)
for cat in ['STEM', 'Non-STEM']:
    n_d = ct.loc[cat, 'Democrat']
    n_i = ct.loc[cat, 'Independent']
    n_r = ct.loc[cat, 'Republican']
    n_t = ct.loc[cat, 'Total']
    p_d = ct_pct.loc[cat, 'Democrat']
    p_i = ct_pct.loc[cat, 'Independent']
    p_r = ct_pct.loc[cat, 'Republican']
    print(f"{cat:20s} {n_d:>7,} ({p_d:4.1f}%) {n_i:>7,} ({p_i:4.1f}%) {n_r:>7,} ({p_r:4.1f}%) {n_t:>7,}")
print("=" * 72)

# Also show column totals
total_row = ct.sum(axis=0)
print(f"\n{'Total':20s} {total_row['Democrat']:>7,}         {total_row['Independent']:>7,}         {total_row['Republican']:>7,}         {total_row['Total']:>7,}")

# ─────────────────────────────────────────────
# 7.  Granular breakdown: individual MAJOR1 × party3
#     sorted by Republican share descending
# ─────────────────────────────────────────────
adf2 = df_ba.dropna(subset=['party3', 'major1_label']).copy()
# Exclude "Other/Unspecified"
adf2 = adf2[adf2['major1_label'] != "Other / Unspecified"]

ct2 = pd.crosstab(adf2['major1_label'], adf2['party3'])[PARTY_ORDER]
ct2['Total'] = ct2.sum(axis=1)
ct2_pct = ct2.div(ct2['Total'], axis=0) * 100
ct2_pct['Total'] = ct2['Total']
ct2_pct = ct2_pct.sort_values('Republican', ascending=False)

print("\n")
print("=" * 95)
print("TABLE 2 — Individual MAJOR1 Category × 3-Category Party ID")
print("          sorted by Republican share (%) descending")
print("          (bachelor's+ degree holders, code 39 'Other' excluded)")
print("=" * 95)
print(f"{'Major Field':45s} {'Dem%':>7s} {'Ind%':>7s} {'Rep%':>7s} {'N':>7s}  STEM?")
print("-" * 95)
for maj, row in ct2_pct.iterrows():
    # Find the code for this label
    code = next((k for k, v in MAJOR1_LABELS.items() if v == maj), None)
    is_stem = "STEM" if code in STEM_CODES else "     "
    print(f"{maj:45s} {row['Democrat']:>6.1f}% {row['Independent']:>6.1f}% {row['Republican']:>6.1f}% {int(row['Total']):>7,}  {is_stem}")
print("=" * 95)

# ─────────────────────────────────────────────
# 8.  STEM vs non-STEM summary stats
# ─────────────────────────────────────────────
print("\n")
print("KEY FINDINGS:")
stem_rep = ct_pct.loc['STEM', 'Republican']
non_stem_rep = ct_pct.loc['Non-STEM', 'Republican']
stem_dem = ct_pct.loc['STEM', 'Democrat']
non_stem_dem = ct_pct.loc['Non-STEM', 'Democrat']
print(f"  STEM majors:     {stem_dem:.1f}% Dem / {ct_pct.loc['STEM','Independent']:.1f}% Ind / {stem_rep:.1f}% Rep  (N={ct.loc['STEM','Total']:,})")
print(f"  Non-STEM majors: {non_stem_dem:.1f}% Dem / {ct_pct.loc['Non-STEM','Independent']:.1f}% Ind / {non_stem_rep:.1f}% Rep  (N={ct.loc['Non-STEM','Total']:,})")
print(f"\n  Republican share gap (non-STEM minus STEM): {non_stem_rep - stem_rep:+.1f} percentage points")
print(f"  Democrat share gap  (STEM minus non-STEM):  {stem_dem - non_stem_dem:+.1f} percentage points")

