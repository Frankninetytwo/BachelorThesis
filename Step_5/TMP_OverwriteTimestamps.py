import os
import csv
import numpy as np

#
# NOTE:
# This was meant to be a temporary file, but ultimately I decided to not delete it, because
# the timestamp adjustment needs some carification:
# The script ExtractFrameTimestamps.py found in the ../Step_3 folder was only applied on the
# files in Step_3/EstimatedGaze (it contains the gaze estimations for the calibration video that
# was used for method validation and method comparison).
# This script right here, however, does not generate timestamps. Instead it served the purpose of
# overwriting the incorrect SIT video timestamps (resulting from the SIT video's corrupted meta data)
# with timestamps provided to me by my supervisor (apparently he had encountered this issue too when
# applying OpenFace on the SIT videos and then corrected the timestamps).
# 


def get_SIT_video_filenames():
    
    filenames = []

    path_without_filename = 'FeatureExtractionData/L2CS-Net'

    # filenames in
    # ../Step_6/CleanedFeatureExtractionData/L2CS-Net
    # are the same as in
    # ../Step_6/CleanedFeatureExtractionData/MCGaze
    for file_or_folder in os.listdir(path_without_filename):

        path = path_without_filename + '/' + file_or_folder
        
        if os.path.isfile(path) and os.path.splitext(path)[1] == '.csv':
            filenames.append(file_or_folder)

    return filenames

def read_openface_frames_with_timestamps(path):

    openface_frames_with_timestamps = []
    openface_data = None

    with open(path) as csv_file:
        openface_data = list(csv.DictReader(csv_file))

    for row in openface_data:
        frame_with_timestamp = dict()
        frame_with_timestamp['frame'] = row['frame']
        frame_with_timestamp['timestamp in s'] = row['timestamp']
        openface_frames_with_timestamps.append(frame_with_timestamp)

    return openface_frames_with_timestamps

def read_extracted_features(method, filename):

    with open('FeatureExtractionData/' + method + '/' + filename) as csv_file:
        return list(csv.DictReader(csv_file))
    
def write_extracted_features_back_to_file(method, filename, extracted_features_with_adjusted_timestamps):
    with open('TMP_FeatureExtractionDataWithCorrectedTimestamps/' + method + '/' + filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=extracted_features_with_adjusted_timestamps[0].keys())
        
        writer.writeheader()
        
        for elem in extracted_features_with_adjusted_timestamps:
            writer.writerow(elem)



filenames = get_SIT_video_filenames()

for filename in filenames:
    for folder in ['open_face_features_alu_mix_lab', 'open_face_features_hu_home', 'open_face_features_hu_mix_lab']:
        possible_path = 'TMP_OpenFaceTimestamps/' + folder + '/' + filename
        if os.path.isfile(possible_path):

            openface_frames_with_timestamps = read_openface_frames_with_timestamps(possible_path)

            for method in ['L2CS-Net', 'MCGaze']:
                extracted_features = read_extracted_features(method, filename)




                #
                #
                #

                L2CSNet = read_extracted_features('L2CS-Net', filename)
                MCGaze = read_extracted_features('MCGaze', filename)
                if len(L2CSNet) < len(MCGaze):
                    print(filename + ": len(L2CSNet) < len(MCGaze)")
                elif len(L2CSNet) > len(MCGaze):
                    print(filename + ": len(L2CSNet) > len(MCGaze)!!! This is unexpected!")
                #
                #
                #
                #


                
                if len(extracted_features) < len(openface_frames_with_timestamps):
                    print(possible_path + ": My file has LESS rows than MBP team's OpenFace file!!! How could this happen? Program will exit")
                    exit()
                elif len(extracted_features) > len(openface_frames_with_timestamps):
                    #print(possible_path + ': My methods analyzed', len(extracted_features) - len(openface_frames_with_timestamps), "frames more than there are in the MBP team's OpenFace file!")
                        
                    # estimate the remaining timestamps
                    mean_diff_prev_10_timestamps = (float(openface_frames_with_timestamps[-1]['timestamp in s']) - float(openface_frames_with_timestamps[-11]['timestamp in s'])) / 10.0
                    
                    for i in range(0, len(extracted_features) - len(openface_frames_with_timestamps)):
                        extracted_features[len(openface_frames_with_timestamps) + i]['timestamp in s'] = str(round(
                            float(openface_frames_with_timestamps[-1]['timestamp in s']) + float(i+1)*mean_diff_prev_10_timestamps,
                            3
                        ))

                for i in range(0, len(openface_frames_with_timestamps)):

                    if int(openface_frames_with_timestamps[i]['frame']) != int(extracted_features[i]['frame']):
                        print("Frames don't match!!! Program will exit")
                        exit()

                    extracted_features[i]['timestamp in s'] = openface_frames_with_timestamps[i]['timestamp in s']

                write_extracted_features_back_to_file(method, filename, extracted_features)

            break