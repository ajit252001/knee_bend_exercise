import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def main():
    st.title('Physio Pal')

    st.sidebar.subheader('Parameters')
    use_webcam = st.sidebar.checkbox('Use Webcam')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.markdown(' ## Output')
    stframe = st.empty()

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture("C:/Users/Aakancha/Downloads/KneeBendVideo.mp4")  # Update with your path
            tfflie.name = "C:/Users/Aakancha/Downloads/KneeBendVideo.mp4"
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))
    
    # Create VideoWriter
    writer = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        stage = None
        bend_time = 0
        reps = 0
        count_reps = False
        feedback_message = None

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(image, str(angle),
                            tuple(np.multiply(knee, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 140:
                    stage = 'leg straight'
                    count_reps = True
                    t0 = time.time()
                if angle < 140:
                    if stage != "knee bend":
                        t0 = time.time()
                        bend_time = 0
                    stage = "knee bend"
                    bend_time = time.time() - t0
                    if bend_time > 8 and count_reps:
                        reps += 1
                        count_reps = False
                        feedback_message = ""
                    if bend_time < 8:
                        feedback_message = "Keep your knee bend"

            except:
                pass

            cv2.rectangle(image, (0, 0), (690, 54), (0, 0, 0), -1)

            cv2.putText(image, "REP", (15, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(reps), (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, 'Stage', (60, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, 'Timer', (270, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "{:.1f}".format(bend_time), (270, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                        cv2.LINE_AA)

            cv2.putText(image, 'Feedback', (350, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, feedback_message, (350, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            writer.write(image)
            stframe.image(image, use_column_width=True)

        vid.release()
        writer.release()
        cv2.destroyAllWindows()
        st.success('Video is Processed')
        st.stop()

if __name__ == '__main__':
    main()



