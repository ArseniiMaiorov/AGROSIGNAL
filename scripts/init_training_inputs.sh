#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${TRAINING_INPUTS_DIR:-$ROOT_DIR/training_inputs}"

mkdir -p "$TARGET_DIR"

cat > "$TARGET_DIR/README.md" <<'EOF'
# Training Inputs

Put local agronomy files here before running:

```bash
make train-all-in-one
```

Required for yield/scenario training:

- `yield_history.csv`
- `crop_plan.csv`

Required for boundary import only if fields are not already present in the DB:

- `field_boundaries.geojson` or `field_boundaries.gpkg`

Optional enrichments:

- `soil_samples.csv`
- `management_events.csv`
- `weather_daily.csv`
EOF

cat > "$TARGET_DIR/yield_history.csv" <<'EOF'
field_external_id,season_year,crop_code,yield_kg_ha
EOF

cat > "$TARGET_DIR/crop_plan.csv" <<'EOF'
field_external_id,season_year,crop_code
EOF

cat > "$TARGET_DIR/soil_samples.csv" <<'EOF'
field_external_id,sampled_at,texture_class,organic_matter_pct,ph,n_ppm,p_ppm,k_ppm
EOF

cat > "$TARGET_DIR/management_events.csv" <<'EOF'
field_external_id,season_year,event_date,event_type,amount,unit,notes
EOF

cat > "$TARGET_DIR/weather_daily.csv" <<'EOF'
field_external_id,season_year,observed_on,precipitation_mm,gdd,vpd,temperature_mean_c
EOF

echo "Initialized training input templates in: $TARGET_DIR"
