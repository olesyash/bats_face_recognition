import cv2
import os

def load_video(video_path, index):
    # Load the video
    video_name = os.path.join(output_folder, str(index))
    print(video_name)
    video = cv2.VideoCapture(video_path)

    # Create a directory to save the frames
    if not os.path.exists(video_name):
        os.makedirs(video_name)

    # Initialize frame counter
    count = 0

    while True:
        # Read the next frame
        ret, frame = video.read()
        if ret:
            # Save the frame as an image file
            cv2.imwrite(f'{video_name}/frame{count}.jpg', frame)
            count += 1
        else:
            # If there are no more frames, break the loop
            break

    # Release the video capture object
    video.release()


input_folder = r"C:\Users\itama\Downloads\bats_2024"
output_folder = r"C:\olesya\bats_frames"
index = 1
for vidio in os.listdir(input_folder):
    vidio_path = os.path.join(input_folder, vidio)
    print(vidio_path)
    load_video(vidio_path, index)
    index += 1
    if index == 3:
        break
