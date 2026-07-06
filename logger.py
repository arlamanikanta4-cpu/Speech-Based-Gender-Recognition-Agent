import csv
import os
from datetime import datetime


def save_log(file_name, gender, confidence):

    os.makedirs("logs", exist_ok=True)

    log_file = "logs/interaction_log.csv"

    exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="") as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "Date",
                "Input File",
                "Gender",
                "Confidence"
            ])

        writer.writerow([
            datetime.now(),
            file_name,
            gender,
            confidence
        ])
