from nibabel.freesurfer.io import read_annot
from cortech import freesurfer
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path


# OUTPUT DIRECTORY


region_dir = Path("/home/oana/MRE_group_results/region_correlations")
region_dir.mkdir(exist_ok=True)


# LOAD ATLAS


labels = {
    "lh": read_annot(
        freesurfer.HOME / "subjects/fsaverage/label/lh.aparc.a2009s.annot"
    ),
    "rh": read_annot(
        freesurfer.HOME / "subjects/fsaverage/label/rh.aparc.a2009s.annot"
    )
}

region_names = [
    r.decode("utf-8")
    for r in labels["lh"][2]
]

subs = list(subject_data_on_fsaverage.keys())


# PARAMETER PAIRS


pairs = [
    ("stiff", "curvature"),
    ("stiff", "sulcal_depth"),
    ("stiff", "thickness"),
]


# LOOP PAIRS


for x_feat, y_feat in pairs:

    print(f"\n{x_feat} vs {y_feat}")

    rows = []

    # -------------------------------------------------
    # COMPUTE REGION-WISE CORRELATIONS
    # -------------------------------------------------

    for region_idx, region_name in enumerate(region_names):

        if region_name == "unknown":
            continue

        all_x = []
        all_y = []

        for sub in subs:

            xd = subject_data_on_fsaverage[sub].get(x_feat)
            yd = subject_data_on_fsaverage[sub].get(y_feat)

            if xd is None or yd is None:
                continue

            for hemi in ["lh", "rh"]:

                mask = labels[hemi][0] == region_idx

                xv = xd[hemi][mask]
                yv = yd[hemi][mask]

                valid = np.isfinite(xv) & np.isfinite(yv)

                all_x.extend(xv[valid])
                all_y.extend(yv[valid])

        if len(all_x) < 20:
            continue

        r, p = spearmanr(all_x, all_y)

        rows.append({
            "region": region_name,
            "r": r,
            "p": p,
            "n": len(all_x)
        })

    
    # DATAFRAME
    

    df_r = pd.DataFrame(rows)

    # FDR correction
    _, df_r["p_fdr"] = fdrcorrection(
        df_r["p"],
        alpha=0.05
    )

    df_r["significant"] = df_r["p_fdr"] < 0.05

    # Save CSV
    df_r.to_csv(
        region_dir / f"{x_feat}_vs_{y_feat}_regional_stats.csv",
        index=False
    )

    
    # CREATE SURFACE MAPS
    

    for hemi in ["lh", "rh"]:

        vertex_labels = labels[hemi][0]
        n_vertices = len(vertex_labels)

        r_map = np.zeros(n_vertices)
        r_fdr_map = np.zeros(n_vertices)
        p_map = np.zeros(n_vertices)

        for region_idx, region_name in enumerate(region_names):

            if region_name == "unknown":
                continue

            mask = vertex_labels == region_idx

            row = df_r[df_r["region"] == region_name]

            if len(row) == 0:
                continue

            r_val = row["r"].values[0]
            p_val = row["p_fdr"].values[0]
            sig = row["significant"].values[0]

            # all correlations
            r_map[mask] = r_val

            # significance map
            p_map[mask] = -np.log10(p_val + 1e-300)

            # FDR-only map
            if sig:
                r_fdr_map[mask] = r_val

        
        # SAVE MAPS
        

        nib.save(
            nib.freesurfer.mghformat.MGHImage(
                r_map.astype("float32"),
                np.eye(4)
            ),
            region_dir / f"{hemi}.{x_feat}_vs_{y_feat}.regional_r.mgh"
        )

        nib.save(
            nib.freesurfer.mghformat.MGHImage(
                r_fdr_map.astype("float32"),
                np.eye(4)
            ),
            region_dir / f"{hemi}.{x_feat}_vs_{y_feat}.regional_r_fdr.mgh"
        )

        nib.save(
            nib.freesurfer.mghformat.MGHImage(
                p_map.astype("float32"),
                np.eye(4)
            ),
            region_dir / f"{hemi}.{x_feat}_vs_{y_feat}.regional_neglog10p.mgh"
        )

    print("Saved maps")
