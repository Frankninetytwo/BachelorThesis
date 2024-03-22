import csv
import os
from FeatureEngineering import EyeGazeFeatures
from pathlib import Path


def get_SIT_video_filenames(methods):

    filenames = dict()

    for method in methods:
        filenames[method] = {
            'ASC': [],
            'NT': []
        }

    with open('../Step_5/FeatureExtractionData/FilenameToConditionMap.csv') as csv_file:
                    
        file_content = list(csv.DictReader(csv_file))

        for row in file_content:

            condition = 'ASC' if int(row['SITCondition.ASD']) == 1 else 'NT'
            
            for method in methods:
                for filename_expansion in ['_part_2', '_part_3', '_part_4']:
                    
                    filename = row['id'] + filename_expansion + '.csv'

                    if os.path.isfile('../Step_6/CleanedFeatureExtractionData/' + method + '/' + filename):
                        filenames[method][condition].append(filename)
                        break

    return filenames

def get_extracted_features(methods):

    filenames = get_SIT_video_filenames(methods)
    extracted_features = dict()

    for method in methods:
        
        # This dictionary separates data by condition.
        extracted_features[method] = dict()
        
        for condition in filenames[method]:
            # Dictionary with filenames as keys.
            extracted_features[method][condition] = dict()
    
    for method in methods:
        for condition in filenames[method]:
            for filename in filenames[method][condition]:
                with open('../Step_6/CleanedFeatureExtractionData/' + method + '/' + filename) as csv_file:
                    
                    file_content = list(csv.DictReader(csv_file))

                    extracted_features[method][condition][filename] = {
                        'frame': [],
                        'timestamp': [],
                        'yaw': [],
                        'pitch': []
                    }

                    for row in file_content:

                        if int(row['success']) == 0:
                            print('There are still frames in the cleaned data where gaze estimation failed! Program will exit.')
                            exit()

                        extracted_features[method][condition][filename]['frame'].append(
                            int(row['frame'])
                        )
                            
                        extracted_features[method][condition][filename]['timestamp'].append(
                            float(row['timestamp in s'])
                        )

                        extracted_features[method][condition][filename]['yaw'].append(
                            float(row['yaw in radians'])
                        )

                        extracted_features[method][condition][filename]['pitch'].append(
                            float(row['pitch in radians'])
                        )
                    
                        
    return extracted_features

def write_features_to_file(features, method, condition):

    with open('FeatureEngineeringData/' + method + '/' + condition + '.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=features[0].keys())
        
        writer.writeheader()
        
        for elem in features:
            writer.writerow(elem)



if __name__ == '__main__':

    methods = ['L2CS-Net', 'MCGaze']
    extracted_features = get_extracted_features(methods)

    print('count L2CS-Net files:', len(extracted_features['L2CS-Net']['ASC']) + len(extracted_features['L2CS-Net']['NT']))
    print('count MCGaze files:', len(extracted_features['MCGaze']['ASC']) + len(extracted_features['MCGaze']['NT']))

    engineered_features = {
        'L2CS-Net': {
            'ASC': [],
            'NT': []
        },
        'MCGaze': {
            'ASC': [],
            'NT': []
        }
    }

    for method in extracted_features:
        for condition in extracted_features[method]:
            for filename in extracted_features[method][condition]:

                features_from_file = extracted_features[method][condition][filename]


                #
                # BEGIN: test if timestamps are correct now
                #

                #fps_reported_by_opencv = 1.0 / ( features_from_file['timestamp'][1] / (features_from_file['frame'][1]-1) )
                #if not (24.9 < fps_reported_by_opencv < 31):
                #    print(filename + ": opencv reports approx.", fps_reported_by_opencv, "FPS")

                #
                # END: test if timestamps are correct now
                #
                    
                    
                gaze_features = EyeGazeFeatures(
                    features_from_file['yaw'],
                    features_from_file['pitch'],
                    features_from_file['timestamp'],
                    method
                ).run()
                
                engineered_features[method][condition].append({'video': Path(filename).stem, **gaze_features.copy()})

                #print('features_from_file:', features_from_file)
                #print('engineered_features:', engineered_features)

            write_features_to_file(engineered_features[method][condition], method, condition)

