import os
import argparse
from pathlib import Path


#
# This is the timestamp extraction script mentioned in the "Method Validation" section of my thesis.
#
# Purpose of this script:
# To validate the methods I had to relate the estimated gaze angles to the ground truth pixel coordinates on the computer screen that the SIT reports.
# Along with the ground truth screen coordinates where the mouse clicks occurred SIT also saves the time when each mouse click happened.
# Hence, I needed to find the frames that the video shows at these times.
# The timestamps in the .csv files that my feature extraction scripts generate are calculated like OpenFace does it; for videos with constant
# frame rate this timestamp is the point in time when the video shows that frame. However, it appears that SIT outputs a video
# with variable frame rate. For this reason I had to extract the timestamps from the SIT calibration video using ffmpeg.
#
# Also note: This script was ONLY applied on the files that contain the extracted features for the SIT calibration video.
# These files can be found in the ./EstimatedGaze folder. This script, however, was not used to correct the timestamps
# of the files generated when applying the methods on the 164 SIT videos. How these were corrected is explained
# in TMP_OverwriteTimestamps.py found in the ../Step_5 folder.
#


def parse_args():

    parser = argparse.ArgumentParser(description='Writes the frame id with timestamp when it is shown in the video to file')
    
    parser.add_argument(
        '--video',
        dest='video_path',
        help='path of the video to proccess',  
        type=str
        )
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    if not args.video_path:
        print("--video argument is mandatory!")
        exit()

    frame_folder_path = 'FramesOf_' + Path(args.video_path).stem

    if os.path.isdir(frame_folder_path):
        print('Before running this script you need to get rid of the folder \"' + frame_folder_path + '\" first')
        exit()


    #
    # First: Generate the frames using ffmpeg, setting the timestamp as filename.
    #

    os.system('mkdir ' + frame_folder_path)

    # source: https://superuser.com/a/1421195
    os.system('ffmpeg -i {} -vsync 0 -r 1000 -frame_pts true {}/%d.png'.format(args.video_path, frame_folder_path))


    #
    # Second: Read in filenames and write frame number with timestamp to file
    #

    timestamps = []

    for filename in os.listdir(frame_folder_path):
        timestamps.append(int(Path(filename).stem))

    timestamps.sort()

    with open(Path(args.video_path).stem + '_Timestamps.csv', 'w') as f:

        f.write('frame,timestamp\n')
        
        for frame_index, timestamp in enumerate(timestamps):
            f.write(str(frame_index + 1) + ',' + str(timestamp) + '\n')


    # remove temporary frame path again
    os.system('rm -r ' + frame_folder_path)
