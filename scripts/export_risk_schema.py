# SPDX-License-Identifier: Apache-2.0
"""Export FEATURE_SCHEMA dict → schema.json. Her config değişikliğinde çalıştır."""
import json
from pathlib import Path
from wildfire_ml.risk.config import FEATURE_SCHEMA

out_path = Path(__file__).parent.parent / "src" / "wildfire_ml" / "risk" / "schema.json"
out_path.write_text(json.dumps(FEATURE_SCHEMA, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
