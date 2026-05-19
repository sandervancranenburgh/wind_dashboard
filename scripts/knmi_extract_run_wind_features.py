import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from next_day_wind_model.knmi_harmonie import SitePoint, extract_tar_features

TAR_PATH = Path("data/raw/knmi/harmonie_arome_cy43_p1/HARM43_V1_P1_2026051504.tar")

# Use your actual spot/sensor coordinates.
SITE = SitePoint(site="valkenburgsemeer", lat=52.168, lon=4.437)


def main():
    out_dir = Path("data/processed/knmi")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = extract_tar_features(TAR_PATH, SITE)
    df = result.frame.copy()

    legacy_aliases = {
        "u_10m_mps": "u_10m",
        "v_10m_mps": "v_10m",
        "wind_speed_10m_mps": "wind_speed_10m",
        "u_50m_mps": "u_50m",
        "v_50m_mps": "v_50m",
        "wind_speed_50m_mps": "wind_speed_50m",
        "u_100m_mps": "u_100m",
        "v_100m_mps": "v_100m",
        "wind_speed_100m_mps": "wind_speed_100m",
        "u_200m_mps": "u_200m",
        "v_200m_mps": "v_200m",
        "wind_speed_200m_mps": "wind_speed_200m",
        "u_300m_mps": "u_300m",
        "v_300m_mps": "v_300m",
        "wind_speed_300m_mps": "wind_speed_300m",
    }
    for source_col, alias_col in legacy_aliases.items():
        if source_col in df.columns and alias_col not in df.columns:
            df[alias_col] = df[source_col]

    out_path = out_dir / "harmonie_p1_wind_features_2026051504.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(df[[
        "horizon_hr",
        "target_ts",
        "wind_speed_10m_mps",
        "wind_speed_10m_knots",
        "wind_speed_100m_mps",
        "speed_shear_10m_100m",
        "wind_dir_10m",
    ]].head())


if __name__ == "__main__":
    main()
