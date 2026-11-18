                                                                                  YOLO_face_detect-reconize

# The rational of the project
Today, security is an important concern for organizations. Advances in AI-based image processing have made facial recognition increasingly useful for access control. However, some systems use fixed cameras and do not actively follow a person as they move.This project combines facial recognition with a two-axis pan-and-tilt camera system. It follows a person’s face as they move within the camera’s tracking range, keeping the face near the center of the image. A web dashboard allows an operator to view the live camera feed and monitor recognition status in real time. Once the system verifies the person as an authorized user, the operator can press the Unlock button to approve access.Before sending the approval command, the system saves the person’s image and a timestamp in a database. These records provide a history of access approvals for later review.

# Literature Review
Previous research provides a foundation for combining facial recognition, access control, and automatic camera tracking. This project draws on five related studies to develop a system that verifies an authorized user, follows a detected face, and records access approvals.
Redmon et al. (2016) introduced YOLO, a unified object detection approach that predicts bounding boxes and class probabilities using a single neural network. This approach supports real-time detection and provides the foundation for locating targets within camera images. In this project, a YOLO-based face detector identifies the face position used by the pan-and-tilt tracking controller [1].
Zhong et al. (2021) proposed SFace, a facial recognition method designed to improve feature discrimination while reducing overfitting caused by imperfect training data. This research supports the identity-verification component of the project. A pretrained SFace model extracts feature vectors from aligned face images, which are compared with the enrolled owner’s features using cosine similarity [2].
Wu, Peng, and Yu (2023) developed YuNet, a lightweight face detector designed for edge devices with limited computing resources. Their work emphasizes balancing detection accuracy, model size, and processing speed. In this project, YuNet identifies facial landmarks within the face region detected by YOLO. These landmarks support face alignment before SFace extracts the recognition features [3].
Elnozahy, Pari, and Liang (2025) developed a Raspberry Pi-based facial recognition door lock system using computer vision and electronic lock-control hardware. Their research demonstrates the practical integration of facial recognition with access control. The present project addresses a related application, while incorporating operator approval through a web dashboard and recording the approved image with a timestamp in MongoDB [4].
Doloiu et al. (2025) presented a pan-and-tilt face-tracking system that combines YOLO-based detection with PID control. Their system calculates the difference between the detected face position and the image center, then adjusts the servo motors to reduce this error. This principle is closely related to the tracking function of the present project. However, the current implementation uses a simpler image-error controller with a dead zone, bounded angular corrections, and angle limits rather than a complete PID controller [5].
Based on these studies, this project integrates face detection, identity verification, two-axis camera tracking, live web monitoring, and approval recording. The Raspberry Pi 5 performs image processing and application management, while the Raspberry Pi Pico receives angle commands through UART and generates the servo PWM signals. The project therefore focuses on integrating established techniques into a working prototype rather than introducing a new recognition algorithm.
References
[1] Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). “You Only Look Once: Unified, Real-Time Object Detection.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. https://arxiv.org/abs/1506.02640
[2] Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X., and Wen, D. (2021). “SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition.” IEEE Transactions on Image Processing, 30, 2587–2598. https://doi.org/10.1109/TIP.2020.3048632
[3] Wu, W., Peng, H., and Yu, S. (2023). “YuNet: A Tiny Millisecond-level Face Detector.” Machine Intelligence Research, 20, 656–665. https://doi.org/10.1007/s11633-023-1423-y
[4] Elnozahy, S. S. F. A., Pari, S. C., and Liang, L. C. (2025). “Raspberry Pi-Based Face Recognition Door Lock System.” IoT, 6(2), 31. https://doi.org/10.3390/iot6020031
[5] Doloiu, M. D., et al. (2025). “Modeling and Control of a Pan–Tilt Servo System for Face Tracking Using Deep Learning and PID.” Engineering Proceedings, 113(1), 75. https://doi.org/10.3390/engproc2025113075


                                <img width="953" height="418" alt="image" src="https://github.com/user-attachments/assets/a3fbbb00-85d5-4185-9696-8c613f89e815" />

                                                                                  LM2596 Spectification

                                 <img width="642" height="686" alt="image" src="https://github.com/user-attachments/assets/68b2d266-aa94-48c4-b260-f1cc18cf9ac1" />

                                                                                   MG996R Specification

                                <img width="519" height="668" alt="image" src="https://github.com/user-attachments/assets/9b6fdc00-ed37-44d7-986c-28aab5d2af84" />
                                <img width="553" height="510" alt="image" src="https://github.com/user-attachments/assets/da5e8d31-a156-4c5f-ad02-05b728b73fa6" />

                                                                                    EMEET C950 1080P
                                                                                       
                                 <img width="643" height="450" alt="image" src="https://github.com/user-attachments/assets/71f3bf4e-c307-459a-b102-225db2a6b395" />
                                <img width="600" height="496" alt="image" src="https://github.com/user-attachments/assets/8be02f58-b4d7-4626-a13d-01e3e7845b80" />

                                       




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
