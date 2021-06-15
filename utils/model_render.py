'''
Basado en el código de entrenamiento del modelo original en tensorflow de VOCA,
adaptado a funcionalidades en Pytorch, la mayoría de este código pertenece al autor del repositorio original.
Repositorio original: https://github.com/TimoBolkart/voca
'''

import cv2
import numpy as np
import os
import tempfile
import threading

from psbody.mesh import Mesh
from scipy.io import wavfile
from subprocess import call

import torch
from torch import nn
from utils.rendering import render_mesh_helper

class ModelRender:
    def __init__(self, config, batcher):
        self.batcher = batcher
        self.num_render_sequences = config['num_render_sequences']
        self.template_mesh = Mesh(filename=config['template_fname'])

    def render_sequences(self, model, device, out_folder, run_in_parallel=True, data_specifier='validation'):
        print('Render %s sequences' % data_specifier)
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(model, device, out_folder, data_specifier))
            thread.start()
            thread.join()
        else:
            self._render_helper(model, device, out_folder, data_specifier)

    def _render_helper(self, model, device, out_folder, data_specifier='validation'):
            print(f'Render {data_specifier} sequences')
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if data_specifier == 'training':
                raw_audio, processed_audio, vertices, templates, subject_idx = self.batcher.get_training_sequences_in_order(
                    self.num_render_sequences)
                condition_subj_idx = [[idx] for idx in subject_idx]
            elif data_specifier == 'validation':
                raw_audio, processed_audio, vertices, templates, subject_idx = self.batcher.get_validation_sequences_in_order(
                    self.num_render_sequences)
                num_training_subjects = self.batcher.get_num_training_subjects()
                condition_subj_idx = [range(num_training_subjects) for idx in subject_idx]
            else:
                raise NotImplementedError('Unknown data specifier %s' % data_specifier)

            for i_seq in range(len(raw_audio)):
                conditions = condition_subj_idx[i_seq]
                for condition_idx in conditions:
                    condition_subj = self.batcher.convert_training_idx2subj(condition_idx)
                    video_fname = os.path.join(out_folder, '%s_%03d_condition_%s.mp4' % (data_specifier, i_seq, condition_subj))
                    self._render_sequences_helper(model, device, video_fname, raw_audio[i_seq], processed_audio[i_seq], templates[i_seq], vertices[i_seq], condition_idx)

    def _render_sequences_helper(self, model, device, video_fname, seq_raw_audio, seq_processed_audio, seq_template, seq_verts, condition_idx):
        def add_image_text(img, text):
            img = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) // 2
            textY = textsize[1] + 10
            cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        num_frames = seq_verts.shape[0]
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        wavfile.write(tmp_audio_file.name, seq_raw_audio['sample_rate'], seq_raw_audio['audio'])

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (1600, 800), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1600, 800), True)

        with torch.no_grad():
            processed_audio = np.expand_dims(np.stack(seq_processed_audio), -1)
            subject_idx = np.repeat(condition_idx, num_frames)
            face_template = np.repeat(seq_template[np.newaxis,:,:,np.newaxis], num_frames, axis=0)

            processed_audio = torch.from_numpy(processed_audio).type(torch.float32).to(device)
            face_template = torch.from_numpy(face_template).type(torch.float32).to(device)
            subject_idx = torch.from_numpy(subject_idx)

            condition = nn.functional.one_hot(subject_idx, self.batcher.get_num_training_subjects()).to(device)

            model.eval()
            predicted_vertices, _ = model(processed_audio, condition, face_template)
            predicted_vertices = predicted_vertices.cpu().numpy()
            predicted_vertices = np.squeeze(predicted_vertices)
            center = np.mean(seq_verts[0], axis=0)

            for i_frame in range(num_frames):
                gt_img = render_mesh_helper(Mesh(seq_verts[i_frame], self.template_mesh.f), center)
                add_image_text(gt_img, 'Captured data')
                pred_img = render_mesh_helper(Mesh(predicted_vertices[i_frame], self.template_mesh.f), center)
                add_image_text(pred_img, 'VOCA prediction')
                img = np.hstack((gt_img, pred_img))
                writer.write(img)
            writer.release()

            cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
                tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
            call(cmd)