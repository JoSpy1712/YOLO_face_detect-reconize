# Pi5 sender : face recognition and servo tracking


# Role of Each code in Pi5
-  yolo_face.py : main code for control all system.
-  servo_tracking.py : making face-center pan/tilt commands, 0 to 180 degrees.
-  uart_commands.py : writing Uart command, Angle messages, and APPROVED messages.
-  unlock_action.py : save the approval JPEG before send APPROVED massage.
-  mongodb_config.py : load the private MongoDB URI from  .env file.
-  web_server.py and web_state.py : authenticated dashboard and thread-safe state.
-  monitor.html : HTML, CSS, and JavaScript for web dashboard.

# YOLO package 'model.pt'

Install the 'ultralytics' Python package in the project virtual environment; it uses PyTorch
to load . the project face-detector weights from 'model.pt' beside 'yolo_face.py'.

# Yunet and Sface

Install 'YuNet' and 'Sface' model. YuNet detects facial landmarks used to align the cropped face. 
SFace extracts facial feature vectors, which are compared with the enrolled owner's features to verify identity.

# Enable Pi5 header UART before connect to Pico

If header UART0 is not already enabled, edit with 'sudo nano /boot/firmware/config.txt' and add the code 'dtoverlay=uart0-pi5' after [all] 
Connect the configured Pi5 GPIO14 TX (physical pin 8) to Pico GP1 RX (physical pin 2), Pi5 GPIO15 RX (physical pin 10) to Pico GP0 TX (physical pin 1). use 115200 baudrate.
and connect grounds.

# Quick start on Pi5
- install .venv (virtual environment)
- install python pipe : open cv , py serial
- install pytorch,yunet,sface
- open uart --uart-port /dev/ttyAMA0 --uart-baud 115200

# Pi PICO : generate pwm signal to servo and action when approve

# main.py : recieve uart massage then command servo tilt,pan and led on board on when approve.
