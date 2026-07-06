from predictor import GenderPredictor
from report_generator import generate_report
from logger import save_log


class VoiceAnalysisAgent:

    def __init__(self):

        self.predictor = GenderPredictor()

    def analyze(self, file_path):

        gender, confidence = self.predictor.predict(file_path)

        report = generate_report(
            file_path,
            gender,
            confidence
        )

        save_log(
            file_path,
            gender,
            confidence
        )

        return report
