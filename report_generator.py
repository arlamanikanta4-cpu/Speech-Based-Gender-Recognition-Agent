import os
from datetime import datetime


def generate_report(file_name, gender, confidence):

    os.makedirs("reports/analysis_reports", exist_ok=True)

    report_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = f"reports/analysis_reports/report_{report_name}.txt"

    with open(path, "w") as f:

        f.write("VOICE ANALYSIS REPORT\n")

        f.write("=" * 40 + "\n")

        f.write(f"Input File : {file_name}\n")

        f.write(f"Predicted Gender : {gender}\n")

        f.write(f"Confidence : {confidence}%\n")

        f.write(f"Generated : {datetime.now()}\n")

    return {

        "Input File": file_name,

        "Gender": gender,

        "Confidence": str(confidence) + "%",

        "Report Saved": path

    }
