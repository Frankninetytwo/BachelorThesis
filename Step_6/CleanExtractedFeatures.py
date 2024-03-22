import os
import csv
import numpy as np
import math


def get_SIT_video_filenames():
    
    filenames = []

    # filenames in
    # ../Step_5/FeatureExtractionData/L2CS-Net
    # are the same as in
    # ../Step_5/FeatureExtractionData/MCGaze
    path_without_filename = '../Step_5/FeatureExtractionData/L2CS-Net'
    for file_or_folder in os.listdir(path_without_filename):

        path = path_without_filename + '/' + file_or_folder
        
        if os.path.isfile(path) and os.path.splitext(path)[1] == '.csv':
            filenames.append(file_or_folder)

    return filenames

def read_csv_file(path):

    file_content = None

    with open(path) as csv_file:
        file_content = list(csv.DictReader(csv_file))

    # adjust types
    for row in file_content:
        row['frame'] = int(row['frame'])
        row['timestamp in s'] = float(row['timestamp in s'])
        row['success'] = int(row['success'])
        row['yaw in radians'] = float(row['yaw in radians'])
        row['pitch in radians'] = float(row['pitch in radians'])

    return file_content

# "closest" refers to the timestamps! (NOT distance-wise)
def find_closest_neighbors(non_NaN_feature_extraction_data, index, count_neighbors):

    neighbor_candidates = []
    index_below = index - 1
    index_above = index + 1

    # Each loop iteration adds one preceding and one succeeding data point to the neighbor candidates. Hence,
    # after the loop len(neighbor_candidates) equals 2*count_neighbors (unless the data point for which we want to find
    # the closest neighbors represents one of the very first resp. very last frames). Example:
    # For this example let x be the data point that we want to find the closest neighbors for (which is non_NaN_feature_extraction_data[index]).
    # If a gaze estimation method's output was considered invalid for multiple preceding frames of x (such data points are discarded before this
    # function is called), then it can happen that the closest data points (time-wise!) are all succeeding frames of x. So there need to be
    # count_neighbors preceding data points and count_neighbors succeeding data points inside neighbor_candidates.
    # Illustration for count_neighbors = 2:
    # frames:                                            n-2          n-1           n           n+1          n+2
    # corresponding gaze estimations (yaw, pitch):    (NaN, NaN)   (NaN, NaN)   (y_0, p_0)   (y_1, p_1)   (y_2, p_2)
    # The frames n-2 and n-1 would be missing in the non_NaN_feature_extraction_data, hence neighbor_candidates would contain the
    # data points for frames n-4, n-3, n+1 and n+2. If the frames were captured with equal time inbetween, then the closest frames to
    # frame n would both be succeeding frames (the frames n+1 and n+2).
    for i in range(count_neighbors):
        if index_below < 0:
            neighbor_candidates.append({'index': index_above, 'time': non_NaN_feature_extraction_data[index_above]['timestamp in s']})
            index_above += 1
        elif index_above >= len(non_NaN_feature_extraction_data):
            neighbor_candidates.append({'index': index_below, 'time': non_NaN_feature_extraction_data[index_below]['timestamp in s']})
            index_below -= 1
        else:
            neighbor_candidates.append({'index': index_below, 'time': non_NaN_feature_extraction_data[index_below]['timestamp in s']})
            index_below -= 1
            neighbor_candidates.append({'index': index_above, 'time': non_NaN_feature_extraction_data[index_above]['timestamp in s']})
            index_above += 1

    neighbor_candidates.sort(key=lambda elem: abs(non_NaN_feature_extraction_data[index]['timestamp in s'] - elem['time']))

    return neighbor_candidates[:count_neighbors]

# The parameter "index" specifies which data point of non_NaN_feature_extraction_data shall be tested.
def is_outlier(non_NaN_feature_extraction_data, index):

    # "neighbors" refers to the closest data points time-wise.
    count_neighbors = 3
    # 800 degrees per second according to:
    """
    @inproceedings{
        author = { Viktor Kelkkanen and Markus Fiedler and DavidLindero },
        title = { Bitrate Requirements of Non-Panoramic VR Remote Rendering },
        booktitle = { Proceedings of the 28th ACM International Conference on Multimedia },
        year = { 2020 },
        organization = { Association for Computing Machinery },
        doi = { 10.1145/3394171.3413681 }
    }
    """
    max_velocity_head_rotation = 800.0 / 180 * math.pi
    # Initially I calculated a maximum total velocity as sum of max. head rotation speed and max. saccade speed, but
    # I discarded that idea again. Somebody rotating head and eyes at maximum speed, perfectly synchronous
    # when they're supposed to talk to the SIT actress seems ridiculous (that would be up to 50 degrees per
    # frame at 30 FPS).
    # Since max. head rotation is faster than max. saccade speed I used only the head rotation speed.
    # The fact that for MCGaze only 69 outliers were found in approx. 900.000 gaze estimations shows that
    # the velcity is definitely not set too low when taking "only" max. head rotation velocity into account.
    #max_velocity_saccade = 700.0 / 180 * math.pi

    closest_neighbors = find_closest_neighbors(non_NaN_feature_extraction_data, index, count_neighbors)

    # Indicates how often the data point is further away (radian-wise) from the (time-wise) closest neighbors than possible given the
    # corresponding time difference. Note: A value greater than 0.0 does not necessarily mean that it is an outlier, because
    # there can be an outlier among the closest neighbors.
    outlier_fraction = 0.0

    for neighbor in closest_neighbors:
        
        # How many radians the gaze can move at most in the time that passed between both frames.
        outlier_threshold = abs(
            non_NaN_feature_extraction_data[index]['timestamp in s'] - neighbor['time']
            ) * max_velocity_head_rotation
        
        yaw_diff = abs(non_NaN_feature_extraction_data[index]['yaw in radians'] - non_NaN_feature_extraction_data[neighbor['index']]['yaw in radians'])
        pitch_diff = abs(non_NaN_feature_extraction_data[index]['pitch in radians'] - non_NaN_feature_extraction_data[neighbor['index']]['pitch in radians'])

        if np.linalg.norm([yaw_diff, pitch_diff]) > outlier_threshold:
            outlier_fraction += 1.0 / count_neighbors

    
    # Use 1.0 as numerator to counter the case when there is an outlier among the (time-wise) closest
    # neighbors (which would otherwise cause the program to categorize the currently tested data point as an outlier).
    # The added epsilon is meant to prevent float precision issues.
    return outlier_fraction > 1.0 / count_neighbors + 0.00001

# Return value is an empty list when the file to which
# the parameter feature_extraction_data belongs must be excluded entirely
# from further evaluation.
def clean_feature_extraction_data(feature_extraction_data, filename):

    #
    # First: Exclude NaN gaze angle data points.
    #

    non_NaN_feature_extraction_data = []

    for elem in feature_extraction_data:

        if elem['success'] == 0:
            continue

        non_NaN_feature_extraction_data.append(elem)

    #
    # Second: Exclude outliers.
    #

    cleaned_data = []

    for i in range(len(non_NaN_feature_extraction_data)):

        if is_outlier(non_NaN_feature_extraction_data, i):
            #print(
            #    'outlier with timestamp', non_NaN_feature_extraction_data[i]['timestamp in s'],
            #    'inside non_NaN_feature_extraction_data was excluded for file', filename,
            #    '(yaw =', non_NaN_feature_extraction_data[i]['yaw in radians'],
            #    'and pitch =', non_NaN_feature_extraction_data[i]['pitch in radians'], ')'
            #    )
            continue

        cleaned_data.append(non_NaN_feature_extraction_data[i])

    #
    # Third: Find out if the video/file is to be excluded entirely.
    #

    # If a method's gaze estimations are not valid for this long (in seconds) it will be
    # considered a long nan angle sequence (the NaNs are not present in the cleaned data anymore, but the
    # missing elements can cause huge time gaps).
    consecutive_nan_angle_threshold = 0.25
    # how often the consecutive nan angle threshold is exceeded
    count_long_nan_angle_sequences = 0

    for i in range(1, len(cleaned_data)):

        if cleaned_data[i]['timestamp in s'] - cleaned_data[i-1]['timestamp in s'] > 2.0*consecutive_nan_angle_threshold:

            print(filename, 'excluded (NaN angle sequence exceeded twice the threshold)')            
            return []
        elif cleaned_data[i]['timestamp in s'] - cleaned_data[i-1]['timestamp in s'] > consecutive_nan_angle_threshold:
            
            count_long_nan_angle_sequences += 1
            
            if count_long_nan_angle_sequences > 2:
                print(filename, 'excluded (' + str(count_long_nan_angle_sequences) + ' times NaN angle threshold exceeded)')                
                return []


    return cleaned_data
    
def write_cleaned_data_to_file(path, cleaned_data):

    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=cleaned_data[0].keys())
        
        writer.writeheader()
        
        for elem in cleaned_data:
            writer.writerow(elem)


#
# IMPORTANT NOTE:
# As as be seen in the printed output, different files are excluded for different gaze estimation methods.
# Hence, the files that were excluded for one method need to manually be deleted for other methods as well
# (only then can the t-test results for different methods be compared to each other).
# These files are located in the /CleanedFeatureExtractionData folder. Command to delete them:
# $ rm XF969267_part_2.csv UV663516_part_2.csv VA087261_part_2.csv TZ662131_part_2.csv US366349_part_2.csv GP213121_part_2.csv pre-01-019_part_2.csv pre-01-002_part_2.csv pre-01-008_part_2.csv pre-91-047_part_2.csv SM271483_part_2.csv CU321485_part_2.csv pre-04-001_part_2.csv UG429428_part_2.csv ST268519_part_2.csv

if __name__ == '__main__':

    methods = ['L2CS-Net', 'MCGaze']
    filenames = get_SIT_video_filenames()

    for method in methods:
        print('\n\nstarting with', method)
        for filename in filenames:

            feature_extraction_data_path = '../Step_5/FeatureExtractionData'

            feature_extraction_data = read_csv_file(feature_extraction_data_path + '/' + method + '/' + filename)

            cleaned_data = clean_feature_extraction_data(feature_extraction_data, filename)

            if cleaned_data == []:
                # file must be excluded entirely from further evaluation
                continue

            write_cleaned_data_to_file('CleanedFeatureExtractionData/' + method + '/' + filename, cleaned_data)





#
# Code ends here. The below is just something I used to test the is_outlier() function.
#

# When setting count_neighbors = 4 and varying max_velocity_head_rotation inside
# the function is_outlier() between 0.99 and 1.01, then the below can be used to verify
# that is_outlier() really works as intended.
"""
test_data = [
    {
        'frame': 0,
        'timestamp in s': 0.0,
        'success': 1,
        'yaw in radians': 0.0,
        'pitch in radians': 0.0
    },
    {
        'frame': 1,
        'timestamp in s': 0.1,
        'success': 1,
        'yaw in radians': 0.1,
        'pitch in radians': 0.0
    },
    {
        'frame': 2,
        'timestamp in s': 0.2,
        'success': 1,
        'yaw in radians': 0.1,
        'pitch in radians': 0.1
    },
    {
        'frame': 3,
        'timestamp in s': 0.3,
        'success': 1,
        'yaw in radians': 0.05,
        'pitch in radians': 0.65
    },
    {
        'frame': 4,
        'timestamp in s': 0.4,
        'success': 1,
        'yaw in radians': 0.2,
        'pitch in radians': 0.2
    },
    {
        'frame': 5,
        'timestamp in s': 0.5,
        'success': 1,
        'yaw in radians': 0.3,
        'pitch in radians': 0.2
    },
    {
        'frame': 6,
        'timestamp in s': 0.6,
        'success': 1,
        'yaw in radians': 0.3,
        'pitch in radians': 0.3
    },
    {
        'frame': 7,
        'timestamp in s': 0.7,
        'success': 1,
        'yaw in radians': 0.4,
        'pitch in radians': 0.3
    },
    {
        'frame': 8,
        'timestamp in s': 0.8,
        'success': 1,
        'yaw in radians': 0.4,
        'pitch in radians': 0.4
    },
    {
        'frame': 9,
        'timestamp in s': 0.9,
        'success': 1,
        'yaw in radians': 0.5,
        'pitch in radians': 0.4
    },
    {
        'frame': 10,
        'timestamp in s': 1.0,
        'success': 1,
        'yaw in radians': 0.5,
        'pitch in radians': 0.5
    },
    {
        'frame': 11,
        'timestamp in s': 1.1,
        'success': 1,
        'yaw in radians': 0.6,
        'pitch in radians': 0.5
    }
]

for i in range(len(test_data)):
    print(i, 'is outlier?', is_outlier(test_data, i))
exit()
"""