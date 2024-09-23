import os
import cv2
import numpy as np
from natsort import natsorted
import ffmpeg
from scipy.interpolate import interp1d
from wandb.old.summary import h5py

import image_stuff


def get_video_duration_ffmpeg(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format'])
        return duration
    except ffmpeg.Error as e:
        print(f"Error reading video file with ffmpeg: {e}")
        return None


def read_steering_angle_and_speed(segment_path):
    # Load steering angle and car speed data from the processed log
    car_speed_folder = os.path.join(segment_path, 'processed_log', 'CAN', 'speed')
    car_speed_value_path = os.path.join(car_speed_folder, 'value')
    car_speed_t_path = os.path.join(car_speed_folder, 't')

    steering_angle_folder = os.path.join(segment_path, 'processed_log', 'CAN', 'steering_angle')
    steering_angle_value_path = os.path.join(steering_angle_folder, 'value')
    steering_angle_t_path = os.path.join(steering_angle_folder, 't')

    if os.path.isfile(car_speed_value_path) and os.path.isfile(car_speed_t_path) and \
            os.path.isfile(steering_angle_value_path) and os.path.isfile(steering_angle_t_path):
        car_speed_t = np.load(car_speed_t_path)
        #car_speed = (np.load(car_speed_value_path)[:, 0], car_speed_t - np.ones(car_speed_t.shape[0]) * np.min(car_speed_t))
        steering_angle_t = np.load(steering_angle_t_path)
        #steering_angle = (np.load(steering_angle_value_path)[:, 0], steering_angle_t - np.ones(steering_angle_t.shape[0]) * np.min(steering_angle_t))

        car_speed = (np.load(car_speed_value_path)[:, 0], car_speed_t)
        steering_angle = (np.load(steering_angle_value_path), steering_angle_t)
        return car_speed, steering_angle
    else:
        print(f"Error: Missing data files in {segment_path}")
        return None, None


def read_frame_times(segment_path):
    # Load steering angle and car speed data from the processed log
    frame_times_path = os.path.join(segment_path, 'global_pose', 'frame_times')

    if os.path.isfile(frame_times_path):
        times = np.load(frame_times_path)
        #return times - np.ones(times.shape[0]) * np.min(times)
        return times
    else:
        print(f"Error: Missing Frame Times in {segment_path}")
        return None


def interpolate(timepoints, speed, frame_timestamps):
    interpolator = interp1d(timepoints, speed, kind='linear', fill_value='extrapolate')
    return interpolator(frame_timestamps)


def get_frame_timestamps(fps, num_frames):
    return np.arange(0, num_frames) / fps


def is_image_black(image):
    # Check if the image is empty (not read correctly)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Check if all pixels are zero
    if np.all(image == 0):
        return True
    else:
        return False


output_width = 544
output_height = 136
max_routes = 3


def export_hevc_video_to_h5(hevc_path, car_speed, steering_angle, frame_timestamps, h5_file, route_counter):
    global current_index
    car_speed_val, car_speed_t = car_speed
    steering_angle_val, steering_angle_t = steering_angle

    #cap = cv2.VideoCapture(hevc_path)
    #num_frames = 0
    #while cap.isOpened():
    #    ret, frame = cap.read()
    #    if not ret:
    #        break

    #    num_frames += 1

    interpolated_speed = interpolate(car_speed_t, car_speed_val, frame_timestamps)
    interpolated_steering_angle = interpolate(steering_angle_t, steering_angle_val, frame_timestamps)

    # Load the steering wheel sprite image
    steering_wheel_img = cv2.imread('assets/steering-wheel.png', cv2.IMREAD_UNCHANGED)
    if steering_wheel_img is None:
        print("Error: Could not load steering wheel image.")
        return

    # Get the dimensions of the steering wheel image
    wheel_h, wheel_w, _ = steering_wheel_img.shape


    cap = cv2.VideoCapture(hevc_path)
    frame_index = 0
    if not cap.isOpened():
        print(f"Error opening video stream or file: {hevc_path}")
        return
    while cap.isOpened() and frame_index < frame_timestamps.shape[0]:
        ret, frame = cap.read()
        if not ret:
            break

        percent_crop = 0.33
        crop_amount = int(frame.shape[0] * percent_crop)
        frame = frame[crop_amount:-crop_amount]

        frame_small = cv2.resize(frame, (output_width, output_height))

        speed = interpolated_speed[frame_index]
        angle = interpolated_steering_angle[frame_index]

        frame_index += 1

        if not is_image_black(frame_small):
            write_to_h5(h5_file, frame_small, f'images_{route_counter}')
            write_to_h5(h5_file, speed, f'speed_{route_counter}')
            write_to_h5(h5_file, angle, f'angle_{route_counter}')
            current_index += 1

        cv2.imshow('Video', frame)
        cv2.imshow('Small Video', frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Num frames: {frame_timestamps.shape[0]}')
    print(f'Speed Len: {car_speed_val.shape[0]}, {car_speed_t.shape[0]}')
    print(f'Angle Len: {steering_angle_val.shape[0]}, {steering_angle_t.shape[0]}')
    cap.release()
    cv2.destroyAllWindows()

current_index = 0


def create_h5_datasets(h5_file):
    # Initial estimate of the number of windows; adjust as needed
    initial_windows = 1000  # Adjust this value as needed
    image_shape = (initial_windows, output_height, output_width, 3)  # Zakucano
    other_shape = (initial_windows, 1)
    for i in range(max_routes):
        h5_file.create_dataset(f'images_{i}', shape=(0,) + image_shape[1:], maxshape=(None,) + image_shape[1:], dtype='uint8')
        h5_file.create_dataset(f'speed_{i}', shape=(0,) + other_shape[1:], maxshape=(None,) + other_shape[1:], dtype='float32')
        h5_file.create_dataset(f'angle_{i}', shape=(0,) + other_shape[1:], maxshape=(None,) + other_shape[1:], dtype='float32')


def write_to_h5(h5_file, value, dataset_name):
    if current_index >= h5_file[dataset_name].shape[0]:
        # Extend the dataset size
        h5_file[dataset_name].resize((h5_file[dataset_name].shape[0] + 1), axis=0)
    h5_file[dataset_name][current_index] = value


def process_dataset(dataset_path, h5_file):
    max_segments = 99999
    segment_counter = 0
    global current_index
    #max_routes = 1
    route_counter = 0
    for chunk in natsorted(os.listdir(dataset_path)):
        chunk_path = os.path.join(dataset_path, chunk)
        if os.path.isdir(chunk_path):
            for route_id in natsorted(os.listdir(chunk_path)):
                print(f'Routes proccessed: {route_counter}')
                if route_counter == max_routes:
                    return
                route_path = os.path.join(chunk_path, route_id)
                if os.path.isdir(route_path):
                    for segment in natsorted(os.listdir(route_path)):
                        print(f'Segments proccessed: {segment_counter}')
                        if segment_counter == max_segments:
                            return
                        segment_path = os.path.join(route_path, segment)
                        if os.path.isdir(segment_path):
                            hevc_file = os.path.join(segment_path, 'video.hevc')
                            if os.path.isfile(hevc_file):
                                car_speed, steering_angle = read_steering_angle_and_speed(segment_path)
                                frame_times = read_frame_times(segment_path)
                                if car_speed is not None and steering_angle is not None:
                                    export_hevc_video_to_h5(hevc_file, car_speed, steering_angle, frame_times, h5_file, route_counter)
                                    print(f'done {segment}')
                                    segment_counter += 1
                    route_counter += 1
                    current_index = 0


if __name__ == "__main__":
    dataset_path = 'data_test'

    output_h5_path = 'driving_images_test.h5'

    with h5py.File(output_h5_path, 'w') as h5_file:
        create_h5_datasets(h5_file)

        process_dataset(dataset_path, h5_file)
