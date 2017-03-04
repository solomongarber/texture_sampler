import cv2
import random
import constants


def get_images():
    # for each category
    for category in constants.CATEGORIES:
        for i in range(6):
            # get video name (e.g. water2.mp4)
            video_name = category + str(i) + '.mp4'
            print('Processing: ' + video_name)
            process_video(video_name)


def process_video(video_name):
    # load video using cv2
    video_cap = cv2.VideoCapture(video_name)
    if video_cap.isOpened():
        ret, frame = video_cap.read()
    else:
        ret = False
    # while there's another frame
    i = 0
    j = 0
    while ret:
        if i % constants.FRAMES_TO_WAIT == 0:
            for k in range(constants.IMAGES_PER_FRAME):
                # save several images from frame to local directory
                random_image = get_random_image(frame)
                file_name = video_name.replace('.mp4', '') + '_' + str(j) + '.png'
                cv2.imwrite(file_name, random_image)
            j += 1
        i += 1
        ret, frame = video_cap.read()
    video_cap.release()


def get_random_image(frame):
    frame_height, frame_width, _ = frame.shape
    left_x = random.randrange(0, frame_width - constants.IMAGE_SIZE)
    top_y = random.randrange(0, frame_height - constants.IMAGE_SIZE)
    # get random 64 x 64 x 3 chunk from frame
    return frame[top_y:top_y + constants.IMAGE_SIZE, left_x:left_x + constants.IMAGE_SIZE, :]


get_images()
