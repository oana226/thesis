##### Initial code for inverting the existing transform 
# and applying it to map MRE to anatomical T1. 

from pathlib import Path
import ants

transform_path = Path(
    "/home/oana/UDel/UDELData/U01_UDEL_0001_01_v3/"
    "U01_UDEL_0001_01_register_to_MRE/"
    "U01_UDEL_0001_01_MREreg_RigidTransform.mat"
)

# Load + invert
tx = ants.read_transform(str(transform_path))
tx_inv = tx.invert()

# Write to a working directory
out_dir = Path("/home/oana/MRE_analysis")
out_dir.mkdir(exist_ok=True)

inv_path = out_dir / "U01_UDEL_0001_01_MREreg_RigidTransform_INV.mat"

ants.write_transform(tx_inv, str(inv_path))

print("Saved inverted transform to:")
print(inv_path)

#############

# MRE stiffness path
anat = Path(
    "/home/oana/UDel/UDELData/U01_UDEL_0001_01_V3/U01_UDEL_0001_01_MRE_AP_50Hz/U01_UDEL_0001_01_MRE_AP_50Hz_anat.nii.gz"
)

t1_anat = Path(
    "/home/oana/UDel/UDELData/fsruns/0001/mri/orig.mgz"
)

out_dir = Path("/home/oana/MRE_analysis")
out_dir.mkdir(exist_ok=True)

out_stiff = out_dir / "MRE_anat_in_T1_space.nii.gz"

# Load images
mre_img = ants.image_read(str(anat))
ref_img = ants.image_read(str(t1_anat))

# Apply inverse transform
mre_in_t1 = ants.apply_transforms(
    fixed=ref_img,
    moving=mre_img,
    transformlist=[str(inv_path)],
    interpolator="linear"
)

ants.image_write(mre_in_t1, str(out_stiff))
print("Saved:", out_stiff)
