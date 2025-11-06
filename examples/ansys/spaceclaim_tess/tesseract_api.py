# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import zipfile
from pathlib import Path, WindowsPath
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

# Example spaceclaim .exe and script file Paths
# spaceclaim_exe = "F:\\Ansys installations\\ANSYS Inc\\v241\\scdm\\SpaceClaim.exe"
# spaceclaim_script = "geometry_generation.scscript"  # Relies on being executed in same directory as tesseract_api.py

#
# Schemata
#


class InputSchema(BaseModel):
    """Input schema for bar geometry design and SDF generation."""

    differentiable_bar_parameters: Differentiable[
        Array[
            (None, None),
            Float32,
        ]
    ] = Field(
        description=(
            "Angular positions around the unit circle for the bar geometry. "
            "The shape is (num_bars, 2), where num_bars is the number of bars "
            "and the second dimension has the start then end location of each bar."
            "The final +1 entry represents the two z height coordinates for the cutting plane which combine with "
            "a third fixed coordinate centered on the grid with z = grid_height / 2"
        )
    )

    differentiable_plane_parameters: Differentiable[
        Array[
            (None,),
            Float32,
        ]
    ] = Field(
        description=(
            "Two cutting plane z point heights which combine with a fixed third point "
            "centered on the grid at z = grid_height / 2. "
            "The shape is (2) "
            "The two points are orthognal at the maximum extemts of the grid (+X and +Y)."
        )
    )

    non_differentiable_parameters: Array[
        (None,),
        Float32,
    ] = Field(
        description=(
            "Flattened array of non-differentiable geometry parameters. "
            "The shape is (2), the first float is the maximum height (mm) of the "
            "grid (pre z-plane cutting). The second is the beam thickness (mm)."
        )
    )

    """static_parameters: list[int] = Field(
        description=(
            "List of integers used to construct the geometry. "
            "The first integer is the number of bars."
        )
    )"""

    string_parameters: list[str] = Field(
        description=(
            "Two string parameters for geometry construction. "
            "First str is Path to Spaceclaim executable. "
            "Second str is Path to Spaceclaim Script (.scscript)."
        )
    )


class TriangularMesh(BaseModel):
    """Triangular mesh representation with fixed-size arrays."""

    points: Array[(None, 3), Float32] = Field(description="Array of vertex positions.")
    faces: Array[(None, 3), Float32] = Field(
        description="Array of triangular faces defined by indices into the points array."
    )


class OutputSchema(BaseModel):
    """Output schema for generated geometry and SDF field."""

    mesh: TriangularMesh = Field(
        description="Triangular mesh representation of the geometry"
    )


#
# Helper functions
#


def build_geometry(
    differentiable_bar_parameters: np.ndarray,
    differentiable_plane_parameters: np.ndarray,
    non_differentiable_parameters: np.ndarray,
    string_parameters: list[str],
) -> list[trimesh.Trimesh]:
    """Build a Spaceclaim geometry from the parameters by modifying template
    .scscript.

    Return a TriangularMesh object.
    """
    spaceclaim_exe = Path(string_parameters[0])
    spaceclaim_script = Path(string_parameters[1])

    # TODO: Want to stop using TemporaryDirectory for spaceclaim script
    # and instead use the unique run directory created everytime the
    # tesseract is run (so there is history).
    with TemporaryDirectory() as temp_dir:
        prepped_file_path, output_file = _prep_scscript(
            temp_dir,
            spaceclaim_script,
            differentiable_bar_parameters,
            differentiable_plane_parameters,
            non_differentiable_parameters,
        )
        run_spaceclaim(spaceclaim_exe, prepped_file_path)

    mesh = trimesh.load(output_file)

    return mesh


def _prep_scscript(
    temp_dir: TemporaryDirectory,
    spaceclaim_script: Path,
    differentiable_bar_parameters: np.ndarray,
    differentiable_plane_parameters: np.ndarray,
    non_differentiable_parameters: np.ndarray,
) -> list[str]:
    """Take tesseract inputs and place into a temp .scscript that will
    be used to run Spaceclaim.

    Return the Path location of this script and the output .stl
    """
    # Define output file name and location
    # TODO: Same as before: can we output grid_fin.stl in the tesseract
    # unique run directory instead of cwd?
    cwd = os.getcwd()
    output_file = os.path.join(cwd, "grid_fin.stl")

    prepped_file_path = os.path.join(temp_dir, os.path.basename(spaceclaim_script))
    shutil.copy(spaceclaim_script, prepped_file_path)

    # Define dict used to input params to .scscript
    keyvalues = {}
    keyvalues["__output__"] = output_file
    keyvalues["__params__.z2"] = str(differentiable_plane_parameters[0])
    keyvalues["__params__.z3"] = str(differentiable_plane_parameters[1])
    keyvalues["__params__.height"] = non_differentiable_parameters[0]
    keyvalues["__params__.thickness"] = non_differentiable_parameters[1]

    num_of_bars = len(differentiable_bar_parameters)

    for i in range(num_of_bars):
        keyvalues[f"__params__.s{i + 1}"] = str(differentiable_bar_parameters[i][0])
        keyvalues[f"__params__.e{i + 1}"] = str(differentiable_bar_parameters[i][1])

    _find_and_replace_keys_in_archive(prepped_file_path, keyvalues)

    return [prepped_file_path, output_file]


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


def run_spaceclaim(spaceclaim_exe: Path, spaceclaim_script: Path) -> None:
    """Runs Spaceclaim subprocess with .exe and script Path locations.

    Returns the subprocess return code as a placeholder.
    """
    env = os.environ.copy()
    cmd = str(
        f'"{spaceclaim_exe}" /UseLicenseMode=True /Welcome=False /Splash=False '
        + f'/RunScript="{spaceclaim_script}" /ExitAfterScript=True /Headless=True'
    )

    # TODO: Not very robust, should probably try use some error handling
    # or timeout logic to prevent stalling if Spaceclaim fails.
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
    """Create a Spaceclaim geometry based on input parameters and export
    as a .stl.
    """
    mesh = build_geometry(
        differentiable_bar_parameters=inputs.differentiable_bar_parameters,
        differentiable_plane_parameters=inputs.differentiable_plane_parameters,
        non_differentiable_parameters=inputs.non_differentiable_parameters,
        string_parameters=inputs.string_parameters,
    )

    return OutputSchema(
        mesh=TriangularMesh(
            points=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32),
        )
    )
