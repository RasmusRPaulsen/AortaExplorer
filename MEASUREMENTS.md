# AortaExplorer measurements
**Updated 2/11-2025**

AortaExplorer computes a set of measurements that are explained below. Not all measurements are available for alle [scan FOV types](SCANFOV.md). They are found in the **AortaExplorer_measurements.csv** file that is a result from running AortaExplorer.

## Some notes:
- For two-parts aorta the two parts are typically named **ascending** and **descending** aorta. Sometime the ascending part is named **annulus** (with reference to the aortic annulus).

## Measurements

Some measurements are anatomically relevant while other are more for debug purposes. They are not always ordered in the same order in the CSV file.

The cross sectional areas are typically measured as the maximal area found in a segment of the aorta. For example the **infrarenal_area** that is the maximal cross sectional area in the segment ranging from the top of the iliac arteries to the renal arteries. For each cross section, the maximum and minimum diameters (mm) are computed by measuring the diameters going through the center of mass of the cross section. The **cl_dist** is the position of the cut on the centerline (mostly used for debugging and visualization).

For the tortuosity measurements there is a **geometrical_length** that is the Euclidean distance between two landmarks on the centerline and the **aortic_length** that is the distance along the centerline. The *aortic tortuosity index (ATI)* is defined as the ratio between the aortic and the geometrical length and should be *>=1*.


- **scan_name** : The full name of the NIFTI file, including path and extension.
- **base_name** : The base name of the NIFTI file, excluding path and extension.
- **aorta_explorer_version** : The version of AortaExplorer that produced the measurements.
- **last_error_message** : If any error occured during processing, the last error is listed here.
- **spacing_0** : In-slice spacing (mm)
- **spacing_1** : In-slice spacing (mm)
- **spacing_2** : Between slice spacing (mm)
- **volume_dims_0** : In slice side length (voxels)
- **volume_dims_1** : In slice side length (voxels)
- **volume_dims_2** : Number of slices (voxels)
- **volume_size_0** : Physical slice side length (mm)
- **volume_size_1** : Physical slice side length (mm)
- **volume_size_2** : Physical length of scan (mm)
- **vol_ascending_aorta** : Volume (mm^3) of the ascending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md).
- **vol_descending_aorta** : Volume (mm^3) of the descending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md). 
- **vol_heart_myocardium** : Volume (mm^3) of the myocardium
- **vol_heart_left_atrium** : Volume (mm^3) of the left atrium 
- **vol_heart_left_ventricle** : Volume (mm^3) of the left ventricular blood pool including trabeculae and papilary muscles. 
- **vol_heart_right_atrium** : Volume (mm^3) of the right atrium  
- **vol_heart_right_ventricle** : Volume (mm^3) of the right ventricle
- **annulus_surface_volume** : Volume (mm^3) of the ascending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md). Measured from the surface of the aorta part.
- **annulus_surface_area** : Area (mm^2) of surface of the ascending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md).
- **descending_surface_volume** : Volume (mm^3) of the descending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md). Measured from the surface of the aorta part.
- **descending_surface_area** : Area (mm^2) of surface of the descending part of the aorta in scans where the aorta is cut in two. For example [FOV type 5](SCANFOV.md).
- **calcification_volume** : Volume (mm^3) of the voxels classified as calcification (experimental, not validated)
- **calcification_pure_lumen_volume** : Volume (mm^3) of the pure lumen of the aorta (used in the calcification computation)
- **calcification_ratio** : Calcified volume divided by pure lumen volume (experimental, not validated)
- **avg_hu** : Average Hounsfield unit value in the aorta lumen (HU)
- **std_hu** : Standard deviation of Hounsfield units value in the aorta lumen (HU)
- **med_hu** : Median Hounsfield unit value in the aorta lumen (HU)
- **q99_hu** : Q99% Hounsfield unit value in the aorta lumen (HU)
- **q01_hu** : Q1% Hounsfield unit value in the aorta lumen (HU)
- **tot_vol** : Volume (mm^3) of the pure lumen of the aorta
- **skeleton_avg_hu** : Average Hounsfield unit value in the initial estimate of the lumen skeleton (HU)
- **skeleton_std_hu** : Standard deviation of Hounsfield units value in the initial estimate of the lumen skeleton (HU)	
- **skeleton_med_hu** : Median Hounsfield unit value in the initial estimate of the lumen skeleton (HU)	
- **skeleton_q99_hu** : Q99% Hounsfield unit value in the initial estimate of the lumen skeleton (HU)	
- **skeleton_q01_hu** : Q1% Hounsfield unit value in the initial estimate of the lumen skeleton (HU)	
- **low_thresh** : The Hounsfield unit low threshold used when computing the pure lumen segment.
- **high_thresh** : The Hounsfield unit high threshold used when computing the pure lumen segment.
- **scan_type** : Automatically computed [FOV type](SCANFOV.md)
- **scan_type_desc** : Textual description of the [FOV type](SCANFOV.md)
- **surface_volume** : Volume (mm^3) of the aorta in scans where the aorta is one part. For example [FOV type 1 or 2](SCANFOV.md). Measured from the surface of the aorta part.
- **surface_area** : Area (mm^2) of surface of the aorta in scans where the aorta is one part. For example [FOV type 1 or 2](SCANFOV.md). Measured from the surface of the aorta part.
- **cl_count** : Number of samples on the centerline for computing HU statistics
- **cl_mean** : Average value of Hounsfield units computed on the centerline (HU).
- **cl_std** : Standard deviation of Hounsfield units computed on the centerline (HU).
- **cl_med** : Median value of Hounsfield units computed on the centerline (HU).
- **cl_q01** : Q1% Hounsfield units computed on the centerline (HU).
- **cl_q99** : Q99% Hounsfield units computed on the centerline (HU).
- **img_window** : Automatically computed level/window values for visualization of aorta slices
- **img_level** : Automatically computed level/window values for visualization of aorta slices
- **descending_area** : The max cross sectional lumen area in the *descending* segment of the aorta (mm^2). For a [FOV type 1](SCANFOV.md) it is the segment between the abdomen and the aortic arch. For a [FOV Type 2](SCANFOV.md) it is the segment from the renal point and up. For a [FOV type 5](SCANFOV.md) it is from the abdomen and up to the end of the visible aorta.
- **descending_cl_dist** : see the overall description.
- **descending_cl_min_diameter** : see the overall description.
- **descending_cl_max_diameter** : see the overall description.
- **infrarenal_area** : The max cross sectional lumen area in the *infrarenal* segment of the aorta (mm^2). 
- **infrarenal_cl_dist** : see the overall description.
- **infrarenal_cl_min_diameter** : see the overall description.
- **infrarenal_cl_max_diameter** : see the overall description.
- **lvot_area** : The minimum cross sectional area of the Left Ventricular Outflow Tract (LVOT) (mm^2).
- **lvot_cl_dist** : see the overall description.
- **lvot_cl_min_diameter** : see the overall description.
- **lvot_cl_max_diameter** : see the overall description.
- **sinus_of_valsalva_area** : The max cross sectional area of the Sinus of Valsalva (mm^2).
- **sinus_of_valsalva_cl_dist** : see the overall description.
- **sinus_of_valsalva_cl_min_diameter** : see the overall description.
- **sinus_of_valsalva_cl_max_diameter** : see the overall description.
- **sinotubular_junction_area** : The minimum cross sectional area of the sinotubular junction (mm^2)
- **sinotubular_junction_cl_dist** : see the overall description.
- **sinotubular_junction_cl_min_diameter** : see the overall description.
- **sinotubular_junction_cl_max_diameter** : see the overall description.
- **ascending_area** : The max cross sectional area of the ascending segment (mm^2). For an aorta with the complete arch, this segment is from the sinotubular junction to the Brachiocephalic trunk. For a [FOV type 5](SCANFOV.md) it is from the sinotubular junction to the top of the scan.
- **ascending_cl_dist** : see the overall description.
- **ascending_cl_min_diameter** : see the overall description.
- **ascending_cl_max_diameter** : see the overall description.
- **aortic_arch_area** : The max cross sectional area of the arch. The segment is between the Brachiocephalic trunk and the  subclavian artery (mm^2)
- **aortic_arch_cl_dist** : see the overall description.
- **aortic_arch_cl_min_diameter** : see the overall description.
- **aortic_arch_cl_max_diameter** : see the overall description.
- **abdominal_area** : The max cross sectional lumen area in the *abdominal* segment of the aorta (mm^2). 
- **abdominal_cl_dist** : see the overall description.
- **abdominal_cl_min_diameter** : see the overall description.
- **abdominal_cl_max_diameter** : see the overall description.
- **annulus_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **annulus_aortic_length** : see the overall description.
- **annulus_geometric_length** : see the overall description.
- **ascending_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **ascending_aortic_length** : see the overall description.
- **ascending_geometric_length** : see the overall description.
- **descending_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **descending_aortic_length** : see the overall description.
- **descending_geometric_length** : see the overall description.
- **ib_arch_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **ib_arch_aortic_length** : see the overall description.
- **ib_arch_geometric_length** : see the overall description.
- **infrarenal_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **infrarenal_aortic_length** :  see the overall description.
- **infrarenal_geometric_length** : see the overall description.
- **abdominal_aortic_tortuosity_index** : See the [tortuosity definitions](figs/AortaExplorer-tortuosity.pdf)
- **abdominal_aortic_length** : see the overall description.
- **abdominal_geometric_length** : see the overall description.
- **sac_aorta_lumen_volume** : Used for a rough estimate of the presence of an aortic aneurysmic sac.
- **sac_original_lumen_volume** : Used for a rough estimate of the presence of an aortic aneurysmic sac.
- **sac_aorta_ratio** : Used for a rough estimate of the presence of an aortic aneurysmic sac.
- **sac_q95_distance** : Used for a rough estimate of the presence of an aortic aneurysmic sac.
- **sac_calcification_volume** : Used for a rough estimate of the presence of an aortic aneurysmic sac.
