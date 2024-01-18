# Drone with Face Recognition, Voice Recognition, and Hand Gesture Control

## Introduction

Welcome to our Drone project, where we have integrated advanced features like face recognition, voice recognition, and hand gesture control to enhance the capabilities and interactivity of the drone.

## Features

1. **Face Recognition:** The drone utilizes a sophisticated face recognition system to identify and track human faces, opening up possibilities for various applications.
2. **Voice Recognition:** Users can effortlessly control the drone using voice commands, providing a hands-free and intuitive mode of operation.
3. **Hand Gesture Control:** The drone interprets hand gestures for navigation and commands, enabling users to interact with simple hand movements.

## Getting Started

### Installation

1. Clone the repository to your local machine.

    ```bash
    git clone https://github.com/Myat-Min-Paing/tellodrone.git
    ```

2. Navigate to the project directory.

    ```bash
    cd your-drone-project
    ```

3. Install dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate   # For Linux/macOS
    # Or
    venv\Scripts\activate      # For Windows
    pip install -r requirements.txt
    ```

### Usage

1. Connect the drone hardware and ensure proper configuration.

2. Run the application.

    - **Face Recognition:**
        ```bash
        python face-detect.py
        ```

    - **Hand Gesture Control:**
        ```bash
        python hand_sign.py
        ```

    - **Voice Recognition:**
        ```bash
        python voice_reco_g.py
        ```

3. Access the drone's control interface through the provided URL.

Follow the on-screen instructions for face recognition, hand gesture control, and voice recognition.

## Contributing

If you would like to contribute to the project, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to express our gratitude to all contributors and the open-source community for their invaluable support.

Happy droning!

*Your Project Team*
