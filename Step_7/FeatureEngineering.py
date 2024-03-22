# This script is based on: https://github.com/mbp-lab/acii23_sit_autism_detection/blob/master/feature_engineering/eye_gaze.py.
# I (Frank Schilling) changed most of the code. Here are some reasons why these changes were necessary:
# 1. The original script used absolute mean gaze error of a method as fixation threshold. However, when applying MethodValidation.ipynb
#    to OpenFace (just include OpenFace into the list "methods" at the very top of the jupyter notebook, right after the imports) it
#    will show very small deviations for the same ground truth, despite its absolute mean gaze error being huge (~9 degrees). So
#    the absolute mean gaze error is not always a good choice for the fixation threshold.
# 2. compute_saccade_amplitude did not represent the feature saccade amplitude as it is described in the literature, because
#    it calculated the euclidean distances between gaze estimations for all two consecutive frames - even if the eyes were fixating something
#    at that point in time without any saccade taking place. It is worth noting that what is calcuated now does still not exactly match
#    the definition of "saccade amplitude", because changes in gaze direction that stem from head pose movements are considered saccade as well.
#    Neither of the methods I used head pose information, so I couldn't distinguish between head and eye ball movement.
# 3. compute_velocity_acceleration used to calculate values even for fixation periods. This messed up the gaze feature mean velocity.
#
# Furthermore, there are now 6 new features implemented:
# 1. and 2.: Mean and standard deviation of pitch gaze angle (careful with height bias when evaluating these two featured!)
# 3. and 4.: Mean and standard deviation of saccade duration (changes in gaze direction that stem from head movements
#            are considered saccade as well, see explanation for saccade amplitude above)
# 5. and 6.: Correlation of fixation duration with pitch resp. yaw gaze angle


import numpy as np
import math


# IMPORTANT:
# __init__() parameters "gaze_angle_x" resp. "gaze_angle_y" must not contain nan values!
# When you remove nan value from the above mentioned parameters don't forget to remove the
# corresonding timestamp from parameter "timestamps" as well!
class EyeGazeFeatures:
    def __init__(self, gaze_angle_x, gaze_angle_y, timestamps, gaze_estimation_method):
        self._gaze_angle_x = gaze_angle_x
        self._gaze_angle_y = gaze_angle_y
        self._timestamps = timestamps
        self._method = gaze_estimation_method

        self._features = {}

    def run(self):
        self._add_to_features('angle_x', self._gaze_angle_x)
        # y gaze angle was excluded with the reasoning that men are taller on average than women. However, the y angle might
        # be important in the context of ASC: If people with ASC avoid looking into the eyes then they need to look elsewhere.
        # Since the background is plain blue (nothing interesting to look at there) it seems natural to look at the hair, neck area or
        # clothes of the actress, hence lower y angle - at least in theory, but the explorative mean pitch gaze angle t-test (without taking 
        # measures against height bias) did not find such an effect.
        self._add_to_features('angle_y', self._gaze_angle_y)


        is_fixation = determine_fixations(
            self._gaze_angle_x,
            self._gaze_angle_y,
            self._timestamps,
            self._method
        )


        fixation_durations, fixation_duration_corr_with_pitch, fixation_duration_corr_with_yaw = compute_fixation_durations(
            self._gaze_angle_x,
            self._gaze_angle_y,
            self._timestamps,
            is_fixation
        )

        self._add_to_features('fixation_duration', fixation_durations)
        self._add_to_features('fixation_duration_with_pitch', fixation_duration_corr_with_pitch)
        self._add_to_features('fixation_duration_with_yaw', fixation_duration_corr_with_yaw)


        saccade_durations, saccade_amplitudes = compute_saccades(
            self._gaze_angle_x,
            self._gaze_angle_y,
            self._timestamps, is_fixation
        )

        self._add_to_features('saccade_duration', saccade_durations)
        self._add_to_features('saccade_amplitude', saccade_amplitudes)


        velocities, accelerations = compute_velocity_acceleration(
            self._gaze_angle_x,
            self._gaze_angle_y,
            self._timestamps,
            is_fixation
        )

        self._add_to_features('velocity', velocities)
        self._add_to_features('acceleration', accelerations)


        return self._features

    def _add_to_features(self, name, values):

        if isinstance(values, list):
            mean = np.mean(values)
            std = np.std(values)
            self._features[f'gaze_mean_{name}'] = 0.0 if np.isnan(mean) else mean
            self._features[f'gaze_std_{name}'] = 0.0 if np.isnan(std) else std
        else:
            # This case is needed for correlations.
            self._features[f'gaze_corr_{name}'] = 0.0 if np.isnan(values) else values


def determine_fixations(gaze_angle_x, gaze_angle_y, timestamps, method):

    # is_fixation[i] will be True if the eyes do not move from timestamp[i] to timestamp[i+1], otherwise is_fixation[i] will be False.
    is_fixation = [True for i in range(0, len(timestamps) - 1)]

    # Refer to Method Validation or Feature Engineering section of my thesis to find out where these values come from.
    threshold_by_method = {
        'L2CS-Net': {
            ### mean + 3*sd = (1.827, 2.363)
            ### mean + 4*sd = (2.276, 2.927)
            ### mean + 5*sd = (2.725, 3.492)
            ### mean + 6*sd = (3.173, 4.056)
            # ---> mean + 7*sd = (3.622, 4.621)
            ### mean + 8*sd = (4.07, 5.186)
            ### mean + 9*sd = (4.519, 5.75)
            ### mean + 10*sd = (4.968, 6.315)
            'yaw': np.radians(3.622),
            'pitch': np.radians(4.621)
        },
        'MCGaze': {
            ### mean + 3*sd = (2.615, 3.305)
            # ---> mean + 4*sd = (3.21, 4.147)
            ### mean + 5*sd = (3.804, 4.989)
            ### mean + 6*sd = (4.399, 5.832)
            ### mean + 7*sd = (4.993, 6.674)
            ### mean + 8*sd = (5.587, 7.516)
            ### mean + 9*sd = (6.182, 8.359)
            ### mean + 10*sd = (6.776, 9.201)
            'yaw': np.radians(3.21),
            'pitch': np.radians(4.147)
        }
    }

    fixation_start_index = 0

    # is_fixation[0] will always be True, even if in reality there is saccade in the beginning.
    for i in range(2, len(timestamps)):

        # The outcommented code below can be used to take into account a 3rd frame when calculating the mean.
        # This reduces the probability that fixation threshold gets crossed by chance despite fixating still taking place. The downside
        # of using this code snippet is that very short fixations won't be detected as such.
        #if is_fixation[i-2]:
        #    if timestamps[i] - timestamps[fixation_start_index] < 0.07:
        #        # Fixation has only just begun. It's unlikely to end within 70 ms so take another
        #        # gaze estimation into account first to prevent that threshold gets crossed by chance due to gaze estimation deviations
        #        # for same ground truth coordinates.
        #        continue

        # If eyes might still be moving calculate difference between gaze estimation of the two consective frames. Otherwise make use of the
        # mean of gaze estimations for current fixation.
        dx = abs(gaze_angle_x[i] - np.mean(gaze_angle_x[fixation_start_index:i])) if is_fixation[i-2] else abs(gaze_angle_x[i] - gaze_angle_x[i-1])
        dy = abs(gaze_angle_y[i] - np.mean(gaze_angle_y[fixation_start_index:i])) if is_fixation[i-2] else abs(gaze_angle_y[i] - gaze_angle_y[i-1])

        if dx >= threshold_by_method[method]['yaw'] or dy >= threshold_by_method[method]['pitch']:
            is_fixation[i-1] = False
        # When elif is evaluated then fixation is happening from timestamps[i-1] to timestamps[i]
        elif not is_fixation[i-2]:
            # From timestamps[i-1] to timestamps[i] eyes did not move, but from timestamps[i-2] to timestamps[i-1] they did move,
            # hence fixation starts.
            fixation_start_index = i-1

    return is_fixation


# Call determine_fixations first to get the parameter is_fixation.
# This function also returns the correlation of fixation durations with the corresponding mean pitch resp. yaw angle in this time period.
# The correlation with yaw angle was just added for the reason "why not?". The correlation with pitch angle was of interest
# as people with ASC might display smaller fixation durations while looking the actress into the eyes (or maybe the face in general).
# When not looking into the eyes or the face then pitch angle probably changes to look down at the actress' pullover as there
# is nothing interesting to look at in the SIT background.
def compute_fixation_durations(gaze_angle_x, gaze_angle_y, timestamps, is_fixation):

    fixation_durations = []
    mean_pitch_angle_during_fixations = []
    mean_yaw_angle_during_fixations = []

    i = 0

    while i < len(is_fixation):
        if is_fixation[i]:
            # Fixation happening from timestamps[i] to timestamps[i+1]
            for j in range(i+1, len(is_fixation)):
                if not is_fixation[j]:
                    # From timestamps[j-1] to timestamps[j] there was still fixation going on, but
                    # from timestamps[j] to timestamps[j+1] the eyes moved.
                    fixation_durations.append(timestamps[j] - timestamps[i])
                    mean_pitch_angle_during_fixations.append(np.mean(gaze_angle_y[i:j+1]))
                    mean_yaw_angle_during_fixations.append(np.mean(gaze_angle_x[i:j+1]))
                    
                    i = j
                    break
                elif j+1 == len(is_fixation):
                    # Video ends with fixation.
                    fixation_durations.append(timestamps[j+1] - timestamps[i])
                    mean_pitch_angle_during_fixations.append(np.mean(gaze_angle_y[i:j+2]))
                    mean_yaw_angle_during_fixations.append(np.mean(gaze_angle_x[i:j+2]))
                    
                    i = j
                    break

        i += 1

    return fixation_durations, np.corrcoef(fixation_durations, mean_pitch_angle_during_fixations)[0][1], np.corrcoef(fixation_durations, mean_yaw_angle_during_fixations)[0][1]


# Call determine_fixations first to get the parameter is_fixation.
# This function (compute_saccades) will assume that saccade happened whenever fixation was not happening.
# Hence, smooth pursuit (there is no reason to smooth pursue anything during the SIT) as well as moving the entire head to
# change gaze direction will be considered saccade as well. As mentioned in one of the comments above neither of the
# methods I used provided head pose information, so it couldn't be differentiated between head rotation and eye ball movement.
def compute_saccades(gaze_angle_x, gaze_angle_y, timestamps, is_fixation):

    saccade_durations = []
    saccade_amplitudes = []

    i = 0

    while i < len(is_fixation):
        if not is_fixation[i]:
            # Eyes start moving from timestamps[i] to timestamps[i+1].
            for j in range(i+1, len(is_fixation)):
                if is_fixation[j]:
                    # From timestamps[j-1] to timestamps[j] the eyes were still moving, but
                    # from timestamps[j] to timestamps[j+1] the eyes stopped moving.
                    saccade_durations.append(timestamps[j] - timestamps[i])
                    saccade_amplitudes.append(math.sqrt( pow(gaze_angle_x[j] - gaze_angle_x[i], 2) + pow(gaze_angle_y[j] - gaze_angle_y[i], 2) ))
                    
                    i = j
                    break
                elif j+1 == len(is_fixation):
                    # Video ends with saccade.
                    saccade_durations.append(timestamps[j+1] - timestamps[i])
                    saccade_amplitudes.append(math.sqrt( pow(gaze_angle_x[j+1] - gaze_angle_x[i], 2) + pow(gaze_angle_y[j+1] - gaze_angle_y[i], 2) ))
                    
                    i = j
                    break

        i += 1

    return saccade_durations, saccade_amplitudes


# Call determine_fixations first to get the parameter is_fixation.
# This function returns a list of velocities and a list of accelerations during saccades. If saccade happens between more than
# 2 consecutive (in case of velocity) resp. more than 3 consecutive (in case of acceleration) frames the list will contain multiple
# velocities resp. accelerations that belong to this saccade. Meaning there are no mean velocities resp. mean accelerations computed.
# This makes sense, because the literature suggests mostly atypical peak velocity in ASC and usually does not mention average
# velocity (which would be: amplitude / saccade duration).
#
# Regarding the accuracy of the computed values it should be noted that the SIT videos have between 24.9 and 30 FPS.
# So roughly one frame all 35 ms. That means saccades mostly end within 1-2 frames, which might lead to
# velocity values with bad precision. The acceleration values will most certainly have very poor accuracy as they are computed
# from values that already have bad precision themselves.
def compute_velocity_acceleration(gaze_angle_x, gaze_angle_y, timestamps, is_fixation):

    gaze_angles = np.array([gaze_angle_x, gaze_angle_y])

    fixation_and_saccade_velocities = np.linalg.norm(np.diff(gaze_angles, axis=1) / np.diff(timestamps), axis=0)
    fixation_and_saccade_accelerations = np.diff(fixation_and_saccade_velocities, axis=0) / np.diff(timestamps[1:])

    # Return only those velocities and accelerations that belong to frames where saccade happened.
    saccade_velocities = []
    saccade_accelerations = []

    for i in range(0, len(is_fixation)):
        if not is_fixation[i]:
            # from timestamps[i] to timestamps[i+1] the eyes moved
            saccade_velocities.append(fixation_and_saccade_velocities[i])

            if i+1 < len(is_fixation) and (not is_fixation[i+1]):
                # from timestamps[i] to timestamps[i+2] the eyes moved
                saccade_accelerations.append(fixation_and_saccade_accelerations[i])

    return saccade_velocities, saccade_accelerations
