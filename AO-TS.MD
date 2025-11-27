# AortaExplorer results on the TotalSegmentator dataset

The [TotalSegmentator training data](https://zenodo.org/records/10047292) is a set with a large variation of cases and pathologies. It has been used to train TotalSegmentator, so of course the baseline segmentations are based on the training data. Still some valuable lessons can be learned.

Here is an example of how to run AortaExplorer on a more complex case:

**CASE s0161**: A *FOV Type 2* with low contrast in the aorta, so we set the minimum Hounsfield value to 40 instead of the default and the *minimum max value* to 300. It has also been padded with values around -1000 (if you pad CT scans, please use values less than -2000 to avoid confusion with air). To be able to detect the boundaries of the scan, the *out-of-scan-value* is set to -1000:

```
AortaExplorer -i /TotalSegmentatorData/s0161/ct.nii.gz  -o /AortaExplorerData/TotalSegmentatorData_s0161_output/ -oh -1000 -lhu 40 -mhu 300 --verbose
```

