import math

import torch
from natsort import natsorted
from torch.utils.data import Dataset
import os
import natsort


class DrivingSimulatorDataset(Dataset):
    def __init__(self, root_dir, device, img_length, img_vocab_size, exponential_base, past_length, quantization_bins):
        self.root_dir = root_dir
        self.img_vocab_size = img_vocab_size
        self.img_length = img_length
        self.exponential_base = exponential_base
        self.past_length = past_length
        self.quantization_bins = quantization_bins

        self.img_file_list = natsorted([f for f in os.listdir(root_dir) if f.endswith('_img.txt')])
        self.angle_file_list = natsorted([f for f in os.listdir(root_dir) if f.endswith('_angle.txt')])
        self.spd_file_list = natsorted([f for f in os.listdir(root_dir) if f.endswith('_speed.txt')])

        self.data_img = []
        self.data_angle = []
        self.data_spd = []
        self.lengths = []

        for file_name in self.img_file_list:
            file_path = os.path.join(self.root_dir, file_name)
            data = self.load_data_img(file_path)
            self.data_img.append(data)
            data_length = (len(data) // self.img_length) - (exponential_base ** (past_length - 1)) - 1  # For next angle and spd
            self.lengths.append(data_length)
            print(file_path)

        self.max_angle = -9999
        self.min_angle = 9999

        self.max_spd = 0
        self.min_spd = 1000

        for file_name in self.angle_file_list:
            file_path = os.path.join(self.root_dir, file_name)
            data = self.load_data_float(file_path)
            self.data_angle.append(data)
            tmp_min_angle = min(data)
            tmp_max_angle = max(data)

            if tmp_min_angle < self.min_angle:
                self.min_angle = tmp_min_angle
            if tmp_max_angle > self.max_angle:
                self.max_angle = tmp_max_angle
            print(file_path)

        for file_name in self.spd_file_list:
            file_path = os.path.join(self.root_dir, file_name)
            data = self.load_data_float(file_path)
            self.data_spd.append(data)
            tmp_min_spd = min(data)
            tmp_max_spd = max(data)

            if tmp_min_spd < self.min_spd:
                self.min_spd = tmp_min_spd
            if tmp_max_spd > self.max_spd:
                self.max_spd = tmp_max_spd
            print(file_path)

        print(f'Min Spd: {self.min_spd}, Max Spd: {self.max_spd}')
        print(f'Min Angle: {self.min_angle}, Max Angle: {self.max_angle}')

        #self.data_img = torch.tensor(self.data_img).to(device)
        #self.data_angle = torch.tensor(self.data_angle).to(device)
        #self.data_spd = torch.tensor(self.data_spd).to(device)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        video_number = 0
        for length in self.lengths:
            if idx >= length:
                video_number += 1
                idx -= length
            else:
                break

        if idx >= self.lengths[video_number - 1] - 1:
            idx -= 1

        context_idx = idx + self.exponential_base ** (self.past_length - 1)
        input_indices = []
        for i in range(0, self.past_length):
            temp_index = context_idx - (self.exponential_base ** i)
            scaled_temp_index = temp_index * self.img_length
            current_data_img_range = self.data_img[video_number][scaled_temp_index:scaled_temp_index + self.img_length][::-1]
            input_indices.extend(current_data_img_range)

        angles = self.data_angle[video_number][idx+1:context_idx+1]
        spds = self.data_spd[video_number][idx+1:context_idx+1]

        angle_indices = [self.quantize_angle(angle) for angle in angles]
        spd_indices = [self.quantize_spd(spd) for spd in spds]

        input_indices = input_indices[::-1]
        input_indices.extend(angle_indices)
        input_indices.extend(spd_indices)
        input_indices_tensor = torch.tensor(input_indices)

        scaled_context_img = context_idx * self.img_length
        context_indices = self.data_img[video_number][scaled_context_img:scaled_context_img + self.img_length]
        context_indices.append(self.quantize_angle(self.data_angle[video_number][context_idx + 1]))
        #context_indices.append(self.quantize_spd(self.data_spd[video_number][context_idx + 1]))
        context_indices.insert(0, self.img_vocab_size + self.quantization_bins * 2)  # Start token
        context_indices_tensor = torch.tensor(context_indices)

        target_indices = self.data_img[video_number][scaled_context_img:scaled_context_img + self.img_length]
        target_indices.append(self.quantize_angle(self.data_angle[video_number][context_idx + 1]))
        target_indices.append(self.quantize_spd(self.data_spd[video_number][context_idx + 1]))
        target_indices_tensor = torch.tensor(target_indices)

        # One-hot encode the sequences
        one_hot_target = torch.nn.functional.one_hot(target_indices_tensor, num_classes=self.img_vocab_size
                                                     + self.quantization_bins*2).float()

        return input_indices_tensor, context_indices_tensor, one_hot_target

    def getitem_for_generation(self, start_idx, generated_imgs, generated_angles, generated_spds):
        video_number = 0

        for length in self.lengths:
            if start_idx >= length:
                video_number += 1
                start_idx -= length
            else:
                break

        if start_idx >= self.lengths[video_number - 1] - 1:
            start_idx -= 1

        context_idx = start_idx + self.exponential_base ** (self.past_length - 1)

        scaled_start_idx = start_idx * self.img_length
        scaled_context_idx = context_idx * self.img_length
        new_img_sequence = self.data_img[video_number][scaled_start_idx:scaled_context_idx + self.img_length]
        for i in range(len(generated_imgs) // self.img_length):
            new_img_sequence.extend(generated_imgs[i * self.img_length:(i+1) * self.img_length])

        new_context_idx = len(new_img_sequence) // self.img_length
        new_start_idx = len(generated_imgs) // self.img_length
        input_indices = []
        for i in range(0, self.past_length):
            temp_index = new_context_idx - (self.exponential_base ** i)
            scaled_temp_index = temp_index * self.img_length
            current_data_img_range = new_img_sequence[scaled_temp_index:scaled_temp_index + self.img_length][::-1]
            input_indices.extend(current_data_img_range)

        angles = self.data_angle[video_number][start_idx+1:context_idx+1]
        angles = angles + [self.dequantize_angle(angle) for angle in generated_angles]
        angles = angles[len(generated_angles):]

        spds = self.data_spd[video_number][start_idx+1:context_idx+1]
        spds = spds + [self.dequantize_spd(spd) for spd in generated_spds]
        spds = spds[len(generated_spds):]

        angle_indices = [self.quantize_angle(angle) for angle in angles]
        spd_indices = [self.quantize_spd(spd) for spd in spds]

        input_indices = input_indices[::-1]
        input_indices.extend(angle_indices)
        input_indices.extend(spd_indices)
        input_indices_tensor = torch.tensor(input_indices)

        return input_indices_tensor

    def get_input_all(self, start_idx):
        video_number = 0

        for length in self.lengths:
            if start_idx > length:
                video_number += 1
                start_idx -= length
            else:
                break

        if start_idx >= self.lengths[video_number - 1] - 1:
            start_idx -= 1

        context_idx = start_idx + self.exponential_base ** (self.past_length - 1)
        scaled_start_idx = start_idx * self.img_length
        scaled_context_idx = context_idx * self.img_length

        return self.data_img[video_number][scaled_start_idx:scaled_context_idx],\
               [self.quantize_angle(x) for x in self.data_angle[video_number][start_idx+1:context_idx+1]],\
               [self.quantize_spd(x) for x in self.data_spd[video_number][start_idx+1:context_idx+1]]

    def load_data_img(self, file_path):
        data = [int(line) for line in open(file_path, 'r', encoding='ascii', errors='replace') if line.strip()]
        return data

    def load_data_float(self, file_path):
        data = [float(line) for line in open(file_path, 'r', encoding='ascii', errors='replace') if line.strip()]
        return data

    def load_data_old(self, file_path):
        with open(file_path, 'r', encoding='ascii', errors='replace') as file:
            text = file.read().split('\n')

        # Convert text to integers
        data = [int(line) for line in text if line.strip()]  # Convert non-empty lines to integers

        return data

    def quantize_spd(self, spd):
        bin_width = (self.max_spd - self.min_spd) / self.quantization_bins
        bin_index = int((spd - self.min_spd) / bin_width)

        # Ensure bin_index is within the range [0, N-1]
        if bin_index < 0 or bin_index > self.quantization_bins - 1:
            print(f'Speed index out of bounds! {bin_index}')

        bin_index = min(max(bin_index, 0), self.quantization_bins - 1)

        return bin_index + self.img_vocab_size

    def quantize_angle(self, angle):
        bin_width = (self.max_angle - self.min_angle) / self.quantization_bins
        bin_index = int((angle - self.min_angle) / bin_width)

        # Ensure bin_index is within the range [0, N-1]
        if bin_index < 0 or bin_index > self.quantization_bins - 1:
            print(f'Angle index out of bounds! {bin_index}')

        bin_index = min(max(bin_index, 0), self.quantization_bins - 1)

        return bin_index + self.img_vocab_size + self.quantization_bins

    def dequantize_spd(self, quantized_spd):
        bin_index = quantized_spd - self.img_vocab_size
        bin_width = (self.max_spd - self.min_spd) / self.quantization_bins

        # Calculate the center value of the bin
        spd = self.min_spd + bin_index * bin_width + bin_width / 2.0

        # Ensure the dequantized speed is within the original range
        #spd = min(max(spd, self.min_spd), self.max_spd)
        if spd > self.max_spd or spd < self.min_spd:
            print('Speed out of bounds!')

        return spd

    def dequantize_angle(self, quantized_angle):
        bin_index = quantized_angle - self.img_vocab_size - self.quantization_bins
        bin_width = (self.max_angle - self.min_angle) / self.quantization_bins

        # Calculate the center value of the bin
        angle = self.min_angle + bin_index * bin_width + bin_width / 2.0

        # Ensure the dequantized angle is within the original range
        #angle = min(max(angle, self.min_angle), self.max_angle)
        if angle > self.max_angle or angle < self.min_angle:
            print('Angle out of bounds!')

        return angle