"""
Server configuration.

All values can be overridden via environment variables (prefix: VIVADO_API_).
Example .env file (Windows):

    VIVADO_API_VIVADO_PATH=C:/Xilinx/Vivado/2025.1/bin/vivado.bat
    VIVADO_API_WORKSPACE_DIR=C:/vivado_api
    VIVADO_API_API_KEY=mysecrettoken

Note: use forward slashes in paths even on Windows — both Python's pathlib
and Vivado's TCL interpreter handle them correctly on all platforms.
"""

from __future__ import annotations
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Default Vivado executable path differs by platform
_DEFAULT_VIVADO = (
    r"C:/Xilinx/Vivado/2025.1/bin/vivado.bat"
    if sys.platform == "win32"
    else "/tools/Xilinx/Vivado/2025.1/bin/vivado"
)

_DEFAULT_WORKSPACE = (
    r"C:/vivado_api"
    if sys.platform == "win32"
    else "/tmp/vivado_api"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIVADO_API_", env_file=".env")

    # Path to the Vivado executable on the server
    vivado_path: str = Field(
        default=_DEFAULT_VIVADO,
        description="Full path to the Vivado binary (vivado.bat on Windows)",
    )

    # Root directory where all uploaded files and job workspaces are stored
    workspace_dir: str = Field(
        default=_DEFAULT_WORKSPACE,
        description="Root workspace directory for uploads and job artefacts",
    )

    # Optional Bearer token for API authentication.
    # Leave empty ("") to disable authentication entirely.
    api_key: str = Field(
        default="",
        description="Bearer token for API auth; empty string disables auth",
    )

    # Maximum wall-clock seconds allowed for a single synthesis run
    synthesis_timeout: int = Field(default=3600, description="Synthesis timeout in seconds")

    # Maximum wall-clock seconds allowed for a simulation run
    simulation_timeout: int = Field(default=600, description="Simulation timeout in seconds")


settings = Settings()


# ---------------------------------------------------------------------------
# Board catalog
# ---------------------------------------------------------------------------
# Each entry: board_id → dict with 'name', 'part', 'family', 'description'
# Add or remove boards as needed for the target server environment.

BOARD_CATALOG: dict[str, dict] = {
    "basys3": {
        "name": "Basys3",
        "part": "xc7a35tcpg236-1",
        "family": "Artix-7",
        "description": "Digilent Basys3 — Artix-7 35T, 33 280 LUTs",
    },
    "nexys4ddr": {
        "name": "Nexys4 DDR",
        "part": "xc7a100tcsg324-1",
        "family": "Artix-7",
        "description": "Digilent Nexys4 DDR — Artix-7 100T, 101 400 LUTs",
    },
    "nexys_video": {
        "name": "Nexys Video",
        "part": "xc7a200tsbg484-1",
        "family": "Artix-7",
        "description": "Digilent Nexys Video — Artix-7 200T, 215 360 LUTs",
    },
    "pynq_z2": {
        "name": "PYNQ-Z2",
        "part": "xc7z020clg400-1",
        "family": "Zynq-7000",
        "description": "TUL PYNQ-Z2 — Zynq-7020, 53 200 LUTs + dual ARM Cortex-A9",
    },
    "zedboard": {
        "name": "ZedBoard",
        "part": "xc7z020clg484-1",
        "family": "Zynq-7000",
        "description": "Avnet ZedBoard — Zynq-7020, 53 200 LUTs + dual ARM Cortex-A9",
    },
    "kcu105": {
        "name": "KCU105",
        "part": "xcku040-ffva1156-2-e",
        "family": "Kintex UltraScale",
        "description": "Xilinx KCU105 — Kintex UltraScale, 242 400 LUTs",
    },
    "vcu118": {
        "name": "VCU118",
        "part": "xcvu9p-flga2104-2L-e",
        "family": "Virtex UltraScale+",
        "description": "Xilinx VCU118 — Virtex UltraScale+, 1 182 240 LUTs",
    },
}
