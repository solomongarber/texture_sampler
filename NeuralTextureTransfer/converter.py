import os
import subprocess
import constants

def convert(input, output):
    path_to_ffmpeg = '/usr/local/bin/ffmpeg'
    bash_command = [path_to_ffmpeg, '-i', input, output]
    subprocess.call(bash_command)


def convert_all():
    # get all files
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    # if they're AVI, convert to MP4
    for file in files:
        for category in constants.CATEGORIES:
            if file.startswith(category) and file.endswith('.avi'):
                # convert to mp4
                output = file.replace('.avi', '.mp4')
                print('Now creating: ' + output)
                # create .mp4 file with file_name as name
                convert(file, output)


convert_all()