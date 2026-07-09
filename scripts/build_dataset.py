"""
Batch dataset builder — downloads all 51 STS cases from IDC,
runs the existing anatomical validation, and keeps only the ones
that pass the knee hard-gate filter.

Usage:
    python scripts/build_dataset.py [--start N] [--limit N] [--output-dir data/ct_knee]

Logs:
    - Passing cases → data/ct_knee/<NAME>_bone_mask.nii.gz (ready for meshing)
    - Failing cases → data/build_log.json with reason
    - Summary table printed at end
"""

import os
import sys
import json
import subprocess
import argparse
import datetime
import SimpleITK as sitk
from idc_index import IDCClient
from pathlib import Path

# Full STS series map — all 51 patients
ALL_SERIES = {
    "STS_001": "1.3.6.1.4.1.14519.5.2.1.5168.1900.293609116849698550139986038601",
    "STS_002": "1.3.6.1.4.1.14519.5.2.1.5168.1900.213265084688298564549535817201",
    "STS_003": "1.3.6.1.4.1.14519.5.2.1.5168.1900.326006423108798456153429940233",
    "STS_004": "1.3.6.1.4.1.14519.5.2.1.5168.1900.952127023780097934747932279670",
    "STS_005": "1.3.6.1.4.1.14519.5.2.1.5168.1900.379274847602196071565482395253",
    "STS_006": "1.3.6.1.4.1.14519.5.2.1.5168.1900.849567849855022085102637138725",
    "STS_007": "1.3.6.1.4.1.14519.5.2.1.5168.1900.315477836840324582280843038439",
    "STS_008": "1.3.6.1.4.1.14519.5.2.1.5168.1900.847939525316830421968251722427",
    "STS_009": "1.3.6.1.4.1.14519.5.2.1.5168.1900.136001226456621344050523916277",
    "STS_010": "1.3.6.1.4.1.14519.5.2.1.5168.1900.197754415891602187397505258429",
    "STS_011": "1.3.6.1.4.1.14519.5.2.1.5168.1900.896652790055363098816591002588",
    "STS_012": "1.3.6.1.4.1.14519.5.2.1.5168.1900.259441731481975380196382885127",
    "STS_013": "1.3.6.1.4.1.14519.5.2.1.5168.1900.323117268867180306479729866352",
    "STS_014": "1.3.6.1.4.1.14519.5.2.1.5168.1900.117486873080715942502806462928",
    "STS_015": "1.3.6.1.4.1.14519.5.2.1.5168.1900.280974839945591691939678349459",
    "STS_016": "1.3.6.1.4.1.14519.5.2.1.5168.1900.147148200898069708586187792692",
    "STS_017": "1.3.6.1.4.1.14519.5.2.1.5168.1900.841006597986679454940023131048",
    "STS_018": "1.3.6.1.4.1.14519.5.2.1.5168.1900.209544460916235211081549625478",
    "STS_019": "1.3.6.1.4.1.14519.5.2.1.5168.1900.283165018876034064226767213068",
    "STS_020": "1.3.6.1.4.1.14519.5.2.1.5168.1900.809109566107366454700895181525",
    "STS_021": "1.3.6.1.4.1.14519.5.2.1.5168.1900.261330247004115089326337003719",
    "STS_022": "1.3.6.1.4.1.14519.5.2.1.5168.1900.154664832055237836297899153059",
    "STS_023": "1.3.6.1.4.1.14519.5.2.1.5168.1900.280037100827962489641744369764",
    "STS_024": "1.3.6.1.4.1.14519.5.2.1.5168.1900.234834325837523458270924991126",
    "STS_025": "1.3.6.1.4.1.14519.5.2.1.5168.1900.272904213121500367633063763460",
    "STS_026": "1.3.6.1.4.1.14519.5.2.1.5168.1900.148585474812349923193020180241",
    "STS_027": "1.3.6.1.4.1.14519.5.2.1.5168.1900.230490093465082443173443486749",
    "STS_028": "1.3.6.1.4.1.14519.5.2.1.5168.1900.913590972108839969190922752912",
    "STS_029": "1.3.6.1.4.1.14519.5.2.1.5168.1900.411177827296921213037048339723",
    "STS_030": "1.3.6.1.4.1.14519.5.2.1.5168.1900.858374854054542425568274192534",
    "STS_031": "1.3.6.1.4.1.14519.5.2.1.5168.1900.223416171912460543931338467125",
    "STS_032": "1.3.6.1.4.1.14519.5.2.1.5168.1900.225389768618855362748822146086",
    "STS_033": "1.3.6.1.4.1.14519.5.2.1.5168.1900.284196742910356500446890046376",
    "STS_034": "1.3.6.1.4.1.14519.5.2.1.5168.1900.144439911844543278375693492502",
    "STS_035": "1.3.6.1.4.1.14519.5.2.1.5168.1900.272272069210338258600600991202",
    "STS_036": "1.3.6.1.4.1.14519.5.2.1.5168.1900.410140902681190621031507018690",
    "STS_037": "1.3.6.1.4.1.14519.5.2.1.5168.1900.806003955463396110734980428715",
    "STS_038": "1.3.6.1.4.1.14519.5.2.1.5168.1900.128236400784850212969105003036",
    "STS_039": "1.3.6.1.4.1.14519.5.2.1.5168.1900.122982333133160398995070072851",
    "STS_040": "1.3.6.1.4.1.14519.5.2.1.5168.1900.312420712750450209120260897514",
    "STS_041": "1.3.6.1.4.1.14519.5.2.1.5168.1900.774282824211445184724368928687",
    "STS_042": "1.3.6.1.4.1.14519.5.2.1.5168.1900.794918299994114817879846530899",
    "STS_043": "1.3.6.1.4.1.14519.5.2.1.5168.1900.187333476409221178353189652627",
    "STS_044": "1.3.6.1.4.1.14519.5.2.1.5168.1900.106642154655933094490596168714",
    "STS_045": "1.3.6.1.4.1.14519.5.2.1.5168.1900.255802715314982515910413032074",
    "STS_046": "1.3.6.1.4.1.14519.5.2.1.5168.1900.866412410217491992609567995590",
    "STS_047": "1.3.6.1.4.1.14519.5.2.1.5168.1900.294113156667293063949985551175",
    "STS_048": "1.3.6.1.4.1.14519.5.2.1.5168.1900.471801759685785628935323716012",
    "STS_049": "1.3.6.1.4.1.14519.5.2.1.5168.1900.386262602559235766910800931996",
    "STS_050": "1.3.6.1.4.1.14519.5.2.1.5168.1900.314874299920812136414098926497",
    "STS_051": "1.3.6.1.4.1.14519.5.2.1.5168.1900.186653801311189416337649712538",
}


def convert_dicom_to_nifti(dicom_dir: str, out_nifti: str) -> bool:
    """Walk dicom_dir for .dcm files and write a single NIfTI volume."""
    dcm_dirs = []
    for root, dirs, files in os.walk(dicom_dir):
        if any(f.endswith('.dcm') for f in files):
            dcm_dirs.append(root)
    if not dcm_dirs:
        return False
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dirs[0]))
    img = reader.Execute()
    sitk.WriteImage(img, out_nifti)
    return True


def run_segmentation(nifti_path: str, mask_path: str, python_exe: str) -> tuple[bool, str]:
    """Run TotalSegmentator-based segmentation and return (success, reason)."""
    cmd = [python_exe, "src/segmentation/run_ct_segmentation.py",
           "--input", nifti_path, "--output", mask_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, "ok"
    # Extract the human-readable reason from stderr
    for line in result.stderr.split('\n'):
        if "Anatomical validation failed:" in line or "ValueError:" in line or "FAILED:" in line:
            return False, line.strip()
    return False, result.stderr.strip()[-200:] if result.stderr else "unknown error"


def delete_dicom_dir(dicom_dir: str):
    """Remove the raw DICOM directory after successful NIfTI conversion."""
    import shutil
    if os.path.exists(dicom_dir):
        shutil.rmtree(dicom_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and filter all 51 STS cases from IDC")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based) to resume a partial run")
    parser.add_argument("--limit", type=int, default=51, help="Max cases to process in this run")
    parser.add_argument("--output-dir", default="data/ct_knee", help="Output directory for NIfTI files")
    parser.add_argument("--log", default="data/build_log.json", help="JSON log path")
    parser.add_argument("--keep-dicom", action="store_true", help="Keep raw DICOM files after conversion")
    args = parser.parse_args()

    python_exe = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)

    # Load existing log if resuming
    log_path = Path(args.log)
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    client = IDCClient.client()
    cases = list(ALL_SERIES.items())
    cases_to_process = cases[args.start : args.start + args.limit]

    passed, failed, skipped = [], [], []

    for i, (name, uid) in enumerate(cases_to_process, start=args.start + 1):
        print(f"\n[{i}/{len(ALL_SERIES)}] -- {name}")

        nifti_path = os.path.join("data", "raw", f"case01_{name}.nii.gz")
        case_out   = os.path.join("outputs", name)
        mask_path  = os.path.join(case_out, "masks", "bone_mask.nii.gz")
        dicom_dir  = os.path.join(args.output_dir, name)
        os.makedirs(os.path.join(case_out, "masks"), exist_ok=True)

        # Already passed in a previous run
        if os.path.exists(mask_path):
            print(f"  ✓ Already processed — mask exists, skipping.")
            skipped.append(name)
            log[name] = {"status": "pass", "note": "pre-existing mask"}
            continue
            
        # Already failed in a previous run (e.g. non-knee scan)
        if name in log and log[name].get("status") == "fail":
            print(f"  ✓ Already failed previously ({log[name].get('reason', 'unknown')}), skipping.")
            skipped.append(name)
            continue

        # Download DICOM if NIfTI not yet converted
        if not os.path.exists(nifti_path):
            print(f"  ↓ Downloading DICOM...")
            try:
                os.makedirs(dicom_dir, exist_ok=True)
                client.download_dicom_series(seriesInstanceUID=uid, downloadDir=dicom_dir)
                print(f"  → Converting to NIfTI...")
                if not convert_dicom_to_nifti(dicom_dir, nifti_path):
                    reason = "No DICOM files found in download"
                    print(f"  ✗ {reason}")
                    log[name] = {"status": "fail", "reason": reason}
                    failed.append((name, reason))
                    continue
                if not args.keep_dicom:
                    delete_dicom_dir(dicom_dir)
                    print(f"  ✓ Converted, DICOM deleted to save space.")
            except Exception as e:
                reason = f"Download/convert error: {e}"
                print(f"  ✗ {reason}")
                log[name] = {"status": "fail", "reason": reason}
                failed.append((name, reason))
                continue
        else:
            print(f"  ✓ NIfTI already exists, skipping download.")

        # Run anatomical validation via bone_segmentation.py
        print(f"  ⚙  Running segmentation + anatomical gate...")
        success, reason = run_segmentation(nifti_path, mask_path, python_exe)
        if success:
            print(f"  ✓ PASSED — knee anatomy confirmed, mask saved.")
            passed.append(name)
            log[name] = {"status": "pass", "timestamp": datetime.datetime.now().isoformat()}
        else:
            print(f"  ✗ FAILED — {reason}")
            # Delete the raw NIfTI too if it failed, to reclaim disk space
            if os.path.exists(nifti_path):
                os.remove(nifti_path)
                print(f"  → Input NIfTI deleted (non-knee scan).")
            failed.append((name, reason))
            log[name] = {"status": "fail", "reason": reason, "timestamp": datetime.datetime.now().isoformat()}

        # Persist log after every case so a crash doesn't lose progress
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2))

    # Final summary
    print("\n" + "="*60)
    print(f"BATCH COMPLETE")
    print(f"  Passed (knee confirmed):  {len(passed) + len(skipped)}")
    print(f"    - New this run:         {len(passed)}")
    print(f"    - Pre-existing:         {len(skipped)}")
    print(f"  Failed (non-knee/error):  {len(failed)}")
    print(f"  Log saved to: {args.log}")
    if failed:
        print("\nFailed cases:")
        for name, reason in failed:
            print(f"  {name}: {reason[:100]}")
    print("="*60)


if __name__ == "__main__":
    main()
