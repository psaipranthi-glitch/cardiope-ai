import wfdb
import os

output_dir = "data/ecg/raw"
os.makedirs(output_dir, exist_ok=True)

records = [
    "100", "101", "102", "103", "104",
    "105", "106", "107", "108", "109"
]

for record in records:
    print(f"Downloading record {record}...")
    
    wfdb.dl_database(
        "mitdb",
        output_dir,
        records=[record]
    )

print("ECG download complete!")