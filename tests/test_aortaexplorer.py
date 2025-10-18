from aortaexplorer.python_api import aortaexplorer, get_default_parameters


def test_aortaexplorer():
    params = get_default_parameters()

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
    # input_file = "C:/data/AortaExplorer/input/DTU_051.nii.gz"
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
    #input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0004_ct.nii.gz"

    # TYpe 2 with major aneurysm and stents
    # input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0045_ct.nii.gz"
    # params["compute_centerline_from_ts_segmentation"] = False

    # output_folder = "C:/data/AortaExplorer/testoutput/"

    input_file = "C:/data/Abdominal/Totalsegmentator_dataset/totalsegmentator_all_ct/s0019_ct.nii.gz"
    output_folder = "C:/data/AortaExplorer/testoutput/"


    device = "gpu"
    verbose = True
    quiet = False

    success = aortaexplorer(input_file, output_folder, params, device=device, verbose=verbose, quiet=quiet)
    # assert success, "AortaExplorer failed to run successfully"


if __name__ == "__main__":
    test_aortaexplorer()
