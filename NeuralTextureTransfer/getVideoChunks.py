import cv2
import random
import constants
import glob

def get_video_chunks():
    # for each category
    for category in constants.CATEGORIES:
        for video_name in glob.glob(category + '*.mp4'):
            # get video name (e.g. water2.mp4)
            print('Getting video chunks from: ' + video_name)
            process_video(video_name)


def process_video(video_name):
    # load video using cv2
    video_cap = cv2.VideoCapture(video_name)
    if video_cap.isOpened():
        ret, frame = video_cap.read()
    else:
        ret = False
    i = 0
    j = 0
    while ret:
        if i % constants.FRAMES_TO_WAIT == 0:
            result_name = video_name.replace('.mp4', '') + '_' + str(j) + '.mp4'
            write_video_chunk(video_cap, frame, result_name)
        ret, frame = video_cap.read()
        i += 1

    video_cap.release()


def write_video_chunk(video_cap, frame, result_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    chunk = cv2.VideoWriter(result_name, fourcc, constants.FPS, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
    chunk.write(frame)
    frame_height, frame_width, _ = frame.shape
    left_x = get_left_x(frame, frame_width)
    top_y = get_top_y(frame, frame_height)
    for i in range(constants.IMAGES_PER_CHUNK):
        if video_cap.isOpened():
            ret, frame = video_cap.read()
            if ret:
                frame = get_frame_portion(frame, left_x, top_y)
                chunk.write(frame)
            else:
                break


def get_frame_portion(frame, left_x, top_y):
    return frame[top_y:top_y + constants.IMAGE_SIZE, left_x:left_x + constants.IMAGE_SIZE, :]


def get_left_x(frame, frame_width):
    return random.randrange(0, frame_width - constants.IMAGE_SIZE)


def get_top_y(frame, frame_height):
    return random.randrange(0, frame_height - constants.IMAGE_SIZE)