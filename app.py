from voice_agent import VoiceAnalysisAgent

def main():

    print("=" * 50)
    print("VOICE GENDER ANALYSIS AGENT")
    print("=" * 50)

    agent = VoiceAnalysisAgent()

    file_path = input("Enter CSV file path: ")

    result = agent.analyze(file_path)

    print("\nPrediction Result")
    print("-----------------------")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
