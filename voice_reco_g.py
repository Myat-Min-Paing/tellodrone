from vosk import Model, KaldiRecognizer
import pyaudio
from djitellopy import Tello
import time

drone_me = Tello()
drone_me.connect()
drone_me.takeoff()

model = Model(r"/Users/myatminpaing/Downloads/vosk-model-small-ja-0.22")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
listening = False
stream.start_stream()
def voice():
    listening = True
    while listening:
        try:
            data = stream.read(4096)
            if recognizer.AcceptWaveform(data):
                result = recognizer.FinalResult()
                text = result[14:-3]
                print(f"Saying:{text}")
                listening = False
                return text
        except OSError:
            pass
        
        
def controll_drone(voice_command):
    try:
        if "右" in voice_command:
            print("右へ行ってください")
            drone_me.send_rc_control(-20, 0, 0, 0)
        elif "左" in voice_command:
            print("左へ行ってください")
            drone_me.send_rc_control(20, 0, 0, 0)
        elif "右" in voice_command and "左" in voice_command:
            print("Sleep")
            drone_me.send_rc_control(0, 0, 0, 0)
        elif "上" in voice_command:
            print("上へ行ってください")
            drone_me.send_rc_control(0, 0, 20, 0)
        elif "下" in voice_command or "し た" in voice_command:
            print("下へ行ってください")
            drone_me.send_rc_control(0, 0, -20, 0)
        elif "上" in voice_command and "下" in voice_command:
            print("Sleep")
            drone_me.send_rc_control(0, 0, 0, 0)
        elif "前" in voice_command:
            print("前へ行ってください")
            drone_me.send_rc_control(0, 20, 0, 0)
        elif "後ろ" in voice_command:
            print("後ろへ行ってください")
            drone_me.send_rc_control(0, -20, 0, 0)
        elif "前" in voice_command and "後ろ" in voice_command:
            print("Sleep")
            drone_me.send_rc_control(0, 0, 0, 0)
            print("Stop")
        elif "止まれ" in voice_command:
            drone_me.land()
        elif "疲れ" in voice_command:
            drone_me.land()
        else:
            print("Sleep")
            drone_me.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass
    
    
try:
    while True:
        print("Waiting")
        drone_me.send_rc_control(0, 0, 0, 0)
        voice_command = voice()
        controll_drone(voice_command) 
except KeyboardInterrupt:
    print("Land")
    drone_me.land()