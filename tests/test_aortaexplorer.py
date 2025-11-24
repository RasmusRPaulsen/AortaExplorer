import numpy as np
from aortaexplorer.python_api import aortaexplorer, get_default_parameters
from shutil import copyfile

def test_aortaexplorer_calcium_estimator():
    params = get_default_parameters()
    params["num_proc_general"] = 8
    device = "gpu"
    verbose = True
    quiet = False
    input_file = "/data/Data/RAPA/AortaExplorer/calcification_test_files.txt"
    output_folder = "/data/Data/RAPA/AortaExplorer/calcification_tests/"

    input_files = np.loadtxt(input_file, dtype=str)
    # strip path and extensions
    base_names = [f.split("/")[-1].replace(".nii.gz", "").replace(".nii", "") for f in input_files]

    std_mults = [1, 3,  5]
    hu_mins = [200, 300, 400, 600, 800]
    for std_mult in std_mults:
        for hu_min in hu_mins:
            # delete all stats files in output to force recomputation
            for base_name in base_names:
                scan_id = base_names
                stats_folder = f"{output_folder}{scan_id}/statistics/"
                stats_file = f"{stats_folder}aorta_statistics.json"
                try:
                    os.remove(stats_file)
                except FileNotFoundError:
                    pass
                stats_file = f"{stats_folder}aorta_calcification_stats.json"
                try:
                    os.remove(stats_file)
                except FileNotFoundError:
                    pass

            params["aorta_calcification_std_multiplier"] = std_mult
            params["aorta_calcification_min_hu_value"] = hu_min
            params["aorta_min_max_hu_value"] = hu_min
            output_measurements = f"{output_folder}calcification_results_std{std_mult}_min{hu_min}.csv"

            success = aortaexplorer(
                input_file, output_folder, params, device=device, verbose=verbose, quiet=quiet
            )
            copyfile(f"{output_folder}AortaExplorer_measurements", output_measurements)


def test_aortaexplorer():
    params = get_default_parameters()

    params["num_proc_general"] = 2
    # input_file = "C:/data/AortaExplorer/input/"

    # Type 5:
    # input_file = "C:/data/AortaExplorer/input/CFA-PILOT_0000_SERIES0010.nii.gz"

    # Type 2:
    # input_file = "C:/data/AortaExplorer/input/DTU_001.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_010.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_049.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_050.nii.gz"
    # params["compute_centerline_from_ts_segmentation"] = True

    # Type 1:
    #input_file = "C:/data/AortaExplorer/input/DTU_051.nii.gz"
    # params["compare_with_totalsegmentator"] = True

    # input_file = "C:/data/AortaExplorer/input/DTU_053.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_060.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_074i.gz"
    # input_file = "C:/data/AortaExplorer/input/DTU_078.nii.gz"

    # Very low contrast
    # input_file = "C:/data/AortaExplorer/input/DTU_085.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/"
    # output_folder = "C:/data/AortaExplorer/testoutput/"

    # input_file = "/storage/Data/DTU-CGPS-1/NIFTI/CGPS-1_10407_SERIES0019.nii.gz"
    # output_folder = "/data/Data/RAPA/AortaExplorer/output/"

    # input_file = "/data/Data/RAPA/totalsegmentator_all_ct/s0016_ct.nii.gz"
    # output_folder = "/data/Data/RAPA/AortaExplorer/ts_train_output/"

    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0001_ct.nii.gz"
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0002_ct.nii.gz"
    # Iliac arteries not found by ts in s0004
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0004_ct.nii.gz"

    # TYpe 2 with major aneurysm and stents
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0045_ct.nii.gz"
    # params["compute_centerline_from_ts_segmentation"] = False

    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0114_ct.nii.gz"
    # params["compute_centerline_from_ts_segmentation"] = False
    # This is not very robust - since real anatomies might have that value
    # params["out_of_reconstruction_value"] = -1024

    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0109_ct.nii.gz"
    # params["out_of_reconstruction_value"] = -1000

    # Type 2: problems computing CL from TS segmentation
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_4281_SERIES0012.nii.gz"
    input_file = "C:/data/AortaExplorer/input/CGPS-1_6954_SERIES0009.nii.gz"
    params["aorta_calcification_std_multiplier"] = 2
    params["aorta_calcification_min_hu_value"] = 300
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_7246_SERIES0009.nii.gz"

    # Type 5: with old time FOV. Here the descending aorta is cut of by the FOV inside the scan
    # input_file = "C:/data/AortaExplorer/input/LAA-1_0008_SERIES0002.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/LAA-1_0005_SERIES0005.nii.gz"
    # output_folder = "C:/data/AortaExplorer/testoutput/"

    # ImageCAS
    # input_file = "C:/data/AortaExplorer/input/3.img.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/4.img.nii.gz"

    # Type 5: where the aorta touches the side of the scan
    # input_file = "C:/data/AortaExplorer/input/8.img.nii.gz"

    # Public dataset
    # input_file = "C:/data/AortaExplorer/input/D1.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/D2.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/abdominal_lymph_nodes.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/pancreas.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/subject001_CTA.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/subject002_CTA.nii.gz"

    # input_file = "C:/data/AortaSeg24/images/images/subject053_CTA.nii.gz"
    # input_file = "C:/data/AortaSeg24/images/images/subject075_CTA.nii.gz"

    # Very large bisection
    # input_file = "C:/data/AortaSeg24/images/images/subject076_CTA.nii.gz"
    # params["compute_centerline_from_ts_segmentation"] = False
    # params["compare_with_totalsegmentator"] = True

    # input_file = "C:/data/AortaExplorer/input/CGPS-1_10000_SERIES0017.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_10943_SERIES0017.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_11194_SERIES0015_volume_0.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_12230_SERIES0004.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_12306_SERIES0019.nii.gz"
    # input_file = "C:/data/AortaExplorer/input/CGPS-1_12551_SERIES0020.nii.gz"
    # params["compare_with_totalsegmentator"] = True

    # input_file = "C:/data/AVT-Aorta/Dongyang/D1/D1.nrrd"
    # input_file = "C:/data/AVT-Aorta/KiTS/K1/K1.nrrd"
    # input_file = "C:/data/AVT-aorta/Rider/R1 (AD)/R1.nrrd"
    # input_file = "C:/data/AVT-aorta/Rider/R3/R3.nrrd"
    # input_file = "C:/data/AVT-Aorta/KiTS/K10/K10.nrrd"
    # params["hounsfield_unit_offset"] = -1000
    # input_file = "C:/data/AVT-Aorta/Dongyang/All/"
    # input_file = "C:/data/Abdominal/Pancreas-CT/PANCREAS_0003/11-24-2015-PANCREAS0003-Pancreas-02648/"
    # input_file = "C:/data/Abdominal/CTLymphNodes/manifest-IVhUf5Gd7581798897432071977/CT Lymph Nodes/ABD_LYMPH_001/09-14-2014-ABDLYMPH001-abdominallymphnodes-30274/abdominallymphnodes-26828/"

    output_folder = "C:/data/AortaExplorer/testoutput/"

    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0019_ct.nii.gz"
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0864_ct.nii.gz"
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s1390_ct.nii.gz"
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0161_ct.nii.gz"
    # # params["forced_aorta_min_hu_value"] = 40
    # # params["forced_aorta_max_hu_value"] = 300
    # params["aorta_min_hu_value"] = 40
    # params["aorta_min_max_hu_value"] = 300
    # params["out_of_reconstruction_value"] = -1000
    # output_folder = "C:/data/AortaExplorer/testoutput/"

    device = "gpu"
    verbose = True
    quiet = False

    success = aortaexplorer(
        input_file, output_folder, params, device=device, verbose=verbose, quiet=quiet
    )
    # assert success, "AortaExplorer failed to run successfully"


if __name__ == "__main__":
    # test_aortaexplorer()
    test_aortaexplorer_calcium_estimator()
