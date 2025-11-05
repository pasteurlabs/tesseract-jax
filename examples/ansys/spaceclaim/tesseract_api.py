# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import zipfile
from pathlib import Path, WindowsPath
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

# Temporary hardcoded spaceclaim .exe and script files
spaceclaim_exe = "F:\\Ansys installations\\ANSYS Inc\\v241\\scdm\\SpaceClaim.exe"
spaceclaim_script = "geometry_generation.scscript"  # Relies on being executed in same directory as tesseract_api.py

"""
Example dict for 8 beam start (s) and end (e) parameters and two z-plane params

keyvalues_test = {"__params__.z2": "200",
                  "__params__.z3": "600",
                  "__params__.s1": "0",
                  "__params__.s2": "1 * (math.pi / 8)",
                  "__params__.s3": "2 * (math.pi / 8)",
                  "__params__.s4": "3 * (math.pi / 8)",
                  "__params__.s5": "4 * (math.pi / 8)",
                  "__params__.s6": "5 * (math.pi / 8)",
                  "__params__.s7": "6 * (math.pi / 8)",
                  "__params__.s8": "7 * (math.pi / 8)",
                  "__params__.e1": "(0) + math.pi",
                  "__params__.e2": "(1 * (math.pi / 8)) + math.pi",
                  "__params__.e3": "(2 * (math.pi / 8)) + math.pi",
                  "__params__.e4": "(3 * (math.pi / 8)) + math.pi",
                  "__params__.e5": "(4 * (math.pi / 8)) + math.pi",
                  "__params__.e6": "(5 * (math.pi / 8)) + math.pi",
                  "__params__.e7": "(6 * (math.pi / 8)) + math.pi",
                  "__params__.e8": "(7 * (math.pi / 8)) + math.pi"}
"""


class InputSchema(BaseModel):
    grid_parameters: dict = Field(
        description="Parameter dictionary defining location of grid beams and Z cutting plane"
    )


class OutputSchema(BaseModel):
    placeholder_output: str = Field(
        description="A placeholder output as this tesseract creates a .stl."
    )


def _safereplace(filedata: str, key: str, value: str) -> str:
    # ensure double backspace in windows path
    if isinstance(value, WindowsPath):
        value = str(value)
        value = value.replace("\\", "\\\\")
    else:
        value = str(value)
    return filedata.replace(key, value)


def _find_and_replace_keys_in_archive(file: Path, keyvalues: dict) -> None:
    # work on the zip in a temporary directory
    with TemporaryDirectory() as temp_dir:
        # extract zip
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # walk through the extracted files/folders
        for foldername, _subfolders, filenames in os.walk(temp_dir):
            for filename in filenames:
                # read in file
                filepath = os.path.join(foldername, filename)
                try:
                    with open(filepath) as f:
                        filedata = f.read()
                except Exception:
                    filedata = None

                # find/replace
                if filedata:
                    for key, value in keyvalues.items():
                        if value is not None:
                            filedata = _safereplace(filedata, key, value)

                    # write to file
                    with open(filepath, "w") as f:
                        f.write(filedata)

        # write out all files back to zip
        with zipfile.ZipFile(file, "w") as zip_ref:
            for foldername, _subfolders, filenames in os.walk(temp_dir):
                for filename in filenames:
                    filepath = os.path.join(foldername, filename)
                    zip_ref.write(
                        filepath,
                        arcname=os.path.relpath(filepath, start=temp_dir),
                    )


def run_spaceclaim(spaceclaim_exe: Path, spaceclaim_script: Path):
    env = os.environ.copy()
    cmd = str(
        f'"{spaceclaim_exe}" /UseLicenseMode=True /Welcome=False /Splash=False '
        + f'/RunScript="{spaceclaim_script}" /ExitAfterScript=True /Headless=True'
    )

    result = subprocess.run(
        cmd,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    return result.returncode


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Create a Spaceclaim geometry based on input parameters and export as a .stl."""
    cwd = os.getcwd()
    output_file = os.path.join(cwd, "grid_fin.stl")

    keyvalues = inputs.grid_parameters.copy()
    keyvalues["__output__"] = output_file

    with TemporaryDirectory() as temp_dir:
        # Copy spaceclaim template script to temp dir
        copied_file_path = os.path.join(temp_dir, os.path.basename(spaceclaim_script))
        shutil.copy(spaceclaim_script, copied_file_path)

        # Update temp spaceclaim script and use to generate .stl
        update_script = _find_and_replace_keys_in_archive(copied_file_path, keyvalues)
        spaceclaim_result = run_spaceclaim(spaceclaim_exe, copied_file_path)

    return OutputSchema(
        placeholder_output=f"Subprocess return code: {spaceclaim_result}"
    )
