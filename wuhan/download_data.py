#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Data Acquisition for Grand Unified Theory Testing
===========================================================
"""

import os
import sys
import urllib.request
import json

def download_ligo_data():
    """Download LIGO data using GWOSC API"""
    print("\n" + "="*70)
    print("DOWNLOADING LIGO DATA")
    print("="*70)

    try:
        from gwosc.datasets import event_gps
        from gwosc import TimeSeries
        print("OK: GWOSC installed")
    except ImportError:
        print("ERROR: GWOSC not installed")
        print("  Installing: pip install gwosc")
        os.system(f"{sys.executable} -m pip install gwosc")
        from gwosc.datasets import event_gps
        from gwosc import TimeSeries

    events = ["GW150914", "GW151226", "GW170104", "GW170814", "GW190412"]

    os.makedirs("ligo_data", exist_ok=True)

    for event in events:
        print(f"\nDownloading {event}...")
        try:
            gps = event_gps(event)

            # Download H1 (Hanford) data
            ts_H1 = TimeSeries.fetch_open_data(
                'H1', gps-32, gps+32, sample_rate=4096, cache=True
            )

            # Download L1 (Livingston) data
            ts_L1 = TimeSeries.fetch_open_data(
                'L1', gps-32, gps+32, sample_rate=4096, cache=True
            )

            # Save to numpy files
            import numpy as np
            np.savez(
                f"ligo_data/{event}.npz",
                H1_data=ts_H1.value,
                L1_data=ts_L1.value,
                times=ts_H1.times.value,
                gps_time=gps,
                sample_rate=4096
            )


            print(f"  OK: Saved {event}.npz")

        except Exception as e:
            print(f"  ERROR downloading {event}: {e}")

    print("\nOK: LIGO data download complete")

def download_codata():
    """Download/hardcode CODATA 2022 constants"""
    print("\n" + "="*70)
    print("CODATA 2022 CONSTANTS")
    print("="*70)    # CODATA 2022 recommended values
    codata = {
        "speed_of_light_c": {
            "value": 299792458,
            "unit": "m/s",
            "uncertainty": 0,
            "relative_uncertainty": 0
        },
        "planck_constant_h": {
            "value": 6.62607015e-34,
            "unit": "J·s",
            "uncertainty": 0,
            "relative_uncertainty": 0
        },
        "gravitational_constant_G": {
            "value": 6.67430e-11,
            "unit": "m³/(kg·s²)",
            "uncertainty": 1.5e-15,
            "relative_uncertainty": 2.2e-5
        },
        "elementary_charge_e": {
            "value": 1.602176634e-19,
            "unit": "C",
            "uncertainty": 0,
            "relative_uncertainty": 0
        },
        "electron_mass_m_e": {
            "value": 9.1093837015e-31,
            "unit": "kg",
            "uncertainty": 2.8e-40,
            "relative_uncertainty": 3.0e-10
        },
        "fine_structure_alpha": {
            "value": 7.2973525693e-3,
            "unit": "dimensionless",
            "uncertainty": 1.1e-12,
            "relative_uncertainty": 1.5e-10
        },
        "rydberg_constant_R_inf": {
            "value": 10973731.568160,
            "unit": "m⁻¹",
            "uncertainty": 0.000021,
            "relative_uncertainty": 1.9e-12
        },
        "boltzmann_constant_k": {
            "value": 1.380649e-23,
            "unit": "J/K",
            "uncertainty": 0,
            "relative_uncertainty": 0
        }
    }


    with open("codata_2022.json", "w") as f:
        json.dump(codata, f, indent=2)

    print("OK: Saved codata_2022.json")

    for key, val in codata.items():
        print(f"  {key}: {val['value']:.6e} +/- {val['uncertainty']:.2e} {val['unit']}")

def check_existing_data():
    """Check what we already have"""
    print("\n" + "="*70)
    print("CHECKING EXISTING DATA")
    print("="*70)

    paths = {
        "Pan-STARRS": "bigG/bigG/hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_sys-full.txt",
        "Micro-fits": "micro-bot-digest/micro-bot-digest/",
        "HDGL JSON": "hdgl_harmonics_spiral10000_analog_v30/hdgl_spiral10000_v30.json"
    }

    for name, path in paths.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"OK: {name:15} - {size/1024:.1f} KB")
            else:
                count = len([f for f in os.listdir(path) if f.endswith('.txt')])
                print(f"OK: {name:15} - {count} files")
        else:
            print(f"MISSING: {name:15} - NOT FOUND")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRAND UNIFIED THEORY - DATA ACQUISITION")
    print("="*70)

    check_existing_data()
    download_codata()

    print("\n" + "="*70)
    print("READY TO DOWNLOAD LIGO DATA")
    print("="*70)
    print("\nThis will download ~300 MB of gravitational wave data.")
    response = input("Continue? (y/n): ")

    if response.lower() == 'y':
        download_ligo_data()
    else:
        print("\nSkipped LIGO download. Run manually when ready:")
        print("  python download_data.py")
