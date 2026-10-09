"""Load with swift deploy --external_plugins; unmarked requests keep their behavior."""
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.responses import JSONResponse
from swift.infer_engine import TransformersEngine
from swift.pipelines.infer.deploy import SwiftDeploy
from swift.utils import get_logger

from ecospec_kg.selection_decoding_v2 import VERSION, implementation_hashes, install_swift_hooks

install_swift_hooks(TransformersEngine, SwiftDeploy, swift_version=version("ms-swift"), json_response=JSONResponse)
get_logger().info(f"EcoSpec selection decoding loaded: {VERSION}; implementation={implementation_hashes()}")
