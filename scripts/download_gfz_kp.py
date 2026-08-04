"""Download the fixed definitive GFZ Kp source used by icBuilder."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from icbuilder.kp import DEFAULT_KP_PATH, GFZ_KP_QUERY, _utc_datetime64
from icbuilder.kp import validate_gfz_kp, validate_gfz_kp_checksum


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=DEFAULT_KP_PATH)
args = parser.parse_args()

with urlopen(GFZ_KP_QUERY) as response:
    content = response.read()

validate_gfz_kp_checksum(content)
source = json.loads(content)
if source.get("meta") != {
    "license": "CC BY 4.0",
    "source": "GFZ Potsdam",
}:
    raise ValueError("unexpected GFZ source metadata")
validate_gfz_kp(
    _utc_datetime64(source["datetime"]),
    source["Kp"],
    source["status"],
    require_complete_interval=True,
)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(content)
print(f"Wrote {len(source['Kp']):,} definitive Kp values to {args.output}")
