import os
import random
import numpy as np
from datetime import datetime
import tensorflow as tf
import pretty_midi
import librosa
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from model import Generator, Discriminator, Classifier


def get_styles(dataset_train: str, styles = []):
    if '_' in dataset_train:
        dataset_train = dataset_train.rsplit('_', maxsplit=1)
        styles.append(dataset_train[1])
        get_styles(dataset_train[0],styles)
    else:
        styles.append(dataset_train.rsplit('/', maxsplit=1)[1])
    return list(reversed(styles))


styles = get_styles('./data/RnB_bossanova_funk_rock')


def write_pianoroll_save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))
    program_nums = ["Electric Guitar (clean)", "Acoustic Bass", "Drums"]
    is_drum = [False, False, True]
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for idx in range(len(images_with_pause_list)):
        if idx == 0:
            instrument_program = pretty_midi.instrument_name_to_program('Electric Guitar (clean)')
            instrument = pretty_midi.Instrument(program=instrument_program, is_drum=is_drum[idx])
            piano_roll = images_with_pause_list[idx]
            tpp = 60.0 / tempo / float(4)
            threshold = 60.0 / tempo / 4
            phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
            piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
            piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
            piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
            for note_num in range(128):
                start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
                start_time = list(tpp * (start_idx[0].astype(float)))
                end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
                end_time = list(tpp * (end_idx[0].astype(float)))
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                temp_start_time = [i for i in start_time]
                temp_end_time = [i for i in end_time]
                for i in range(len(start_time)):
                    if start_time[i] in temp_start_time and i != len(start_time) - 1:
                        t = []
                        current_idx = temp_start_time.index(start_time[i])
                        for j in range(current_idx + 1, len(temp_start_time)):
                            if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                                t.append(j)
                        for _ in t:
                            temp_start_time.pop(t[0])
                            temp_end_time.pop(t[0])
                start_time = temp_start_time
                end_time = temp_end_time
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                if len(end_time) < len(start_time):
                    d = len(start_time) - len(end_time)
                    start_time = start_time[:-d]
                for idx in range(len(start_time)):
                    if duration[idx] >= threshold:
                        note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx], end=end_time[idx])
                        instrument.notes.append(note)
                    else:
                        if start_time[idx] + threshold <= phrase_end_time:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx], end=start_time[idx] + threshold)
                        else:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx], end=phrase_end_time)
                        instrument.notes.append(note)
            instrument.notes.sort(key=lambda note: note.start)
            midi.instruments.append(instrument)
        if idx == 1:
            instrument_program = pretty_midi.instrument_name_to_program('Acoustic Bass')
            instrument = pretty_midi.Instrument(program=instrument_program, is_drum=is_drum[idx])
            piano_roll = images_with_pause_list[idx]
            tpp = 60.0 / tempo / float(4)
            threshold = 60.0 / tempo / 4
            phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
            piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
            piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
            piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
            for note_num in range(128):
                start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
                start_time = list(tpp * (start_idx[0].astype(float)))
                end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
                end_time = list(tpp * (end_idx[0].astype(float)))
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                temp_start_time = [i for i in start_time]
                temp_end_time = [i for i in end_time]
                for i in range(len(start_time)):
                    if start_time[i] in temp_start_time and i != len(start_time) - 1:
                        t = []
                        current_idx = temp_start_time.index(start_time[i])
                        for j in range(current_idx + 1, len(temp_start_time)):
                            if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                                t.append(j)
                        for _ in t:
                            temp_start_time.pop(t[0])
                            temp_end_time.pop(t[0])
                start_time = temp_start_time
                end_time = temp_end_time
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                if len(end_time) < len(start_time):
                    d = len(start_time) - len(end_time)
                    start_time = start_time[:-d]
                for idx in range(len(start_time)):
                    if duration[idx] >= threshold:
                        note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx], end=end_time[idx])
                        instrument.notes.append(note)
                    else:
                        if start_time[idx] + threshold <= phrase_end_time:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx],
                                                    end=start_time[idx] + threshold)
                        else:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx],
                                                    end=phrase_end_time)
                        instrument.notes.append(note)
            instrument.notes.sort(key=lambda note: note.start)
            midi.instruments.append(instrument)
        if idx == 2:
            instrument = pretty_midi.Instrument(program=0, is_drum=is_drum[idx])
            piano_roll = images_with_pause_list[idx]
            tpp = 60.0 / tempo / float(4)
            threshold = 60.0 / tempo / 4
            phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
            piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
            piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
            piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
            for note_num in range(128):
                start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
                start_time = list(tpp * (start_idx[0].astype(float)))
                end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
                end_time = list(tpp * (end_idx[0].astype(float)))
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                temp_start_time = [i for i in start_time]
                temp_end_time = [i for i in end_time]
                for i in range(len(start_time)):
                    if start_time[i] in temp_start_time and i != len(start_time) - 1:
                        t = []
                        current_idx = temp_start_time.index(start_time[i])
                        for j in range(current_idx + 1, len(temp_start_time)):
                            if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                                t.append(j)
                        for _ in t:
                            temp_start_time.pop(t[0])
                            temp_end_time.pop(t[0])
                start_time = temp_start_time
                end_time = temp_end_time
                duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
                if len(end_time) < len(start_time):
                    d = len(start_time) - len(end_time)
                    start_time = start_time[:-d]
                for idx in range(len(start_time)):
                    if duration[idx] >= threshold:
                        note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx], end=end_time[idx])
                        instrument.notes.append(note)
                    else:
                        if start_time[idx] + threshold <= phrase_end_time:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx],
                                                    end=start_time[idx] + threshold)
                        else:
                            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time[idx],
                                                    end=phrase_end_time)
                        instrument.notes.append(note)
            instrument.notes.sort(key=lambda note: note.start)
            midi.instruments.append(instrument)
    midi.write(file_path)


class StarGAN(object):
    def __init__(self, dataset_loader, args):
        self.args = args
        self.dataset_loader = dataset_loader

        self.cycle_loss_weight = args.cycle_loss_weight
        self.domclass_loss_weight = args.domclass_loss_weight
        self.identity_loss_weight = args.identity_loss_weight
        self.sigma_d = args.sigma_d

        self.dataset_directory = args.dataset_directory
        self.test_directory = args.test_directory
        self.classify_directory = args.classify_directory
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr_decay_epochs = args.lr_decay_epochs
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.classifier_lr = args.classifier_lr
        self.discriminator_updates = args.discriminator_updates
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.resume_epochs = args.resume_epochs

        self.classifier_epochs = args.classifier_epochs
        self.test_epochs = args.test_epochs
        self.target_style = ast.literal_eval(args.target_style)
        self.source_style = args.source_style

        self.use_tensorboard = args.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.styles_encoder = LabelBinarizer().fit(styles)

        self.logs_directory = args.logs_directory
        self.samples_directory = args.samples_directory
        self.models_directory = args.models_directory
        self.results_directory = args.results_directory

        self.log_freq = args.log_freq
        self.sample_freq = args.sample_freq
        self.model_freq = args.model_freq
        self.lr_update_freq = args.lr_update_freq

        self.G = Generator()
        self.D = Discriminator()
        self.C = Classifier()

        self.generator_optimizer = torch.optim.Adam(self.G.parameters(), self.generator_lr, [self.beta1, self.beta2])
        self.discriminator_optimizer = torch.optim.Adam(self.D.parameters(), self.discriminator_lr, [self.beta1, self.beta2])
        self.classifier_optimizer = torch.optim.Adam(self.C.parameters(), self.classifier_lr, [self.beta1, self.beta2])

        # self.generator_optimizer = torch.optim.RMSprop(self.G.parameters(), self.generator_lr)
        # self.discriminator_optimizer = torch.optim.RMSprop(self.D.parameters(), self.discriminator_lr)
        # self.classifier_optimizer = torch.optim.RMSprop(self.C.parameters(), self.classifier_lr)

        # self.generator_optimizer = torch.optim.SGD(self.G.parameters(), self.generator_lr)
        # self.discriminator_optimizer = torch.optim.SGD(self.D.parameters(), self.discriminator_lr)
        # self.classifier_optimizer = torch.optim.SGD(self.C.parameters(), self.classifier_lr)

        # print(self.G)
        # print(self.D)
        # print(self.C)

        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)

        if self.use_tensorboard:
            self.writer = tf.summary.create_file_writer(self.logs_directory)

    def train(self):
        generator_lr = self.generator_lr
        discriminator_lr = self.discriminator_lr
        classifier_lr = self.classifier_lr

        start_epochs = 0

        data = iter(self.dataset_loader)

        print('Training started')
        start_time = datetime.now()

        for i in range(start_epochs, self.epochs):
            try:
                x_real, style_idx_source, label_source = next(data)
            except:
                data = iter(self.dataset_loader)
                x_real, style_idx_source, label_source = next(data)

            gaussian_noise = self.sigma_d * torch.randn(x_real.size())

            rand_idx = torch.randperm(label_source.size(0))
            label_target = label_source[rand_idx]
            style_idx_target = style_idx_source[rand_idx]

            x_real = x_real.to(self.device)
            label_source = label_source.to(self.device)
            label_target = label_target.to(self.device)
            style_idx_source = style_idx_source.to(self.device)
            style_idx_target = style_idx_target.to(self.device)
            gaussian_noise = gaussian_noise.to(self.device)

            cross_entropy_loss = nn.CrossEntropyLoss()
            c_real = self.C(x_real)
            c_loss_real = cross_entropy_loss(input=c_real, target=style_idx_source)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()

            c_loss_real.backward()
            self.classifier_optimizer.step()

            loss = {}
            loss_to_print = {}

            loss['C_loss'] = c_loss_real.item()
            loss_to_print['Classifier loss'] = c_loss_real.item()

            out_real = self.D(x_real + gaussian_noise, label_source)
            x_fake = self.G(x_real, label_target)
            out_fake = self.D(x_fake + gaussian_noise, label_target)
            d_loss_t = F.mse_loss(input=out_fake, target=torch.zeros_like(out_fake, dtype=torch.float)) + F.mse_loss(input=out_real, target=torch.ones_like(out_real, dtype=torch.float))

            out_class = self.C(x_fake)
            d_loss_class = cross_entropy_loss(input=out_class, target=style_idx_target)

            d_loss = d_loss_t + self.domclass_loss_weight * d_loss_class

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()

            d_loss.backward()
            self.discriminator_optimizer.step()

            loss['D_loss'] = d_loss.item()
            loss_to_print['Discriminator loss'] = d_loss.item()

            if (i + 1) % self.discriminator_updates == 0:
                x_fake = self.G(x_real, label_target)
                g_out_source = self.D(x_fake + gaussian_noise, label_target)
                g_loss_fake = F.mse_loss(input=g_out_source, target=torch.ones_like(g_out_source, dtype=torch.float))

                out_class = self.C(x_real)
                g_loss_class = cross_entropy_loss(input=out_class, target=style_idx_source)

                x_reconstituted = self.G(x_fake, label_source)
                g_loss_reconstituted = F.l1_loss(x_reconstituted, x_real)

                x_fake_identity = self.G(x_real, label_source)
                g_loss_fake_id = F.l1_loss(x_fake_identity, x_real)

                g_loss = g_loss_fake + self.cycle_loss_weight * g_loss_reconstituted + self.domclass_loss_weight * g_loss_class + self.identity_loss_weight * g_loss_fake_id

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                self.classifier_optimizer.zero_grad()

                g_loss.backward()
                self.generator_optimizer.step()

                loss['G_loss_fake'] = g_loss_fake.item()
                loss['G_loss_reconstituted'] = g_loss_reconstituted.item()
                loss['G_loss_class'] = g_loss_class.item()
                loss['G_loss_fake_id'] = g_loss_fake_id.item()
                loss['G_loss'] = g_loss.item()

                loss_to_print['Generator loss'] = g_loss.item()

            if (i + 1) % self.log_freq == 0:
                estimated_time = datetime.now() - start_time
                estimated_time = str(estimated_time)[:-7]
                log = f'Elapsed [{estimated_time}] Epochs [{i+1}/{self.epochs}]'
                for tag, value in loss_to_print.items():
                    log += f', {tag}: {value:.4f}'
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        summary = tf.summary.scalar(tag, value, step=i + 1)
                        self.writer.flush()

            if (i + 1) % self.sample_freq == 0:
                with torch.no_grad():
                    if self.source_style:
                        source_style = self.source_style
                    else:
                        source_style = random.choice(styles)
                    files = os.path.join(self.test_directory, source_style)
                    files = librosa.util.find_files(files, ext='npy')
                    npy_files = {}
                    for f in files:
                        filename = os.path.basename(f)
                        file = np.load(f) * 1.
                        if not npy_files.__contains__(filename):
                            npy_files[filename] = {}
                        npy_files[filename] = file
                    label_source_style = self.styles_encoder.transform([source_style])[0]
                    label_source_style = np.asarray([label_source_style])
                    target_style = random.choice([x for x in styles if x != source_style])
                    label_target_style = self.styles_encoder.transform([target_style])[0]
                    label_target_style = np.asarray([label_target_style])
                    for filename, npy in npy_files.items():
                        filename = filename.split('.')[0]
                        npy_mod = torch.FloatTensor(npy).to(self.device)
                        npy_mod = npy_mod.view(1, npy_mod.size(0), npy_mod.size(1), npy_mod.size(2))
                        label_target = torch.FloatTensor(label_target_style)
                        npy_mod = npy_mod.to(self.device)
                        label_target = label_target.to(self.device)
                        npy_transfer = self.G(npy_mod, label_target)
                        label_source = torch.FloatTensor(label_source_style)
                        label_source = label_source.to(self.device)
                        npy_cycle = self.G(npy_transfer.to(self.device), label_source).data.cpu().numpy()
                        npy_transfer = npy_transfer.data.cpu().numpy()
                        track_is_max = np.equal(npy_transfer, np.amax(npy_transfer, axis=-1, keepdims=True))
                        track_pass_threshold = (npy_transfer > 0.5)
                        npy_transfer_binary = np.logical_and(track_is_max, track_pass_threshold)
                        track_is_max_cycle = np.equal(npy_cycle, np.amax(npy_cycle, axis=-1, keepdims=True))
                        track_pass_threshold_cycle = (npy_cycle > 0.5)
                        npy_cycle_binary = np.logical_and(track_is_max_cycle, track_pass_threshold_cycle)
                        npy_transfer_binary = npy_transfer_binary.reshape(-1, npy_transfer_binary.shape[2], npy_transfer_binary.shape[3], npy_transfer_binary.shape[1])
                        npy_cycle_binary = npy_cycle_binary.reshape(-1, npy_cycle_binary.shape[2], npy_cycle_binary.shape[3], npy_cycle_binary.shape[1])
                        name_origin = f'{source_style}-{target_style}_iter{i + 1}_{filename}_origin'
                        name_transfer = f'{source_style}-{target_style}_iter{i + 1}_{filename}_transfer'
                        name_cycle = f'{source_style}-{target_style}_iter{i + 1}_{filename}_cycle'
                        path_samples_epochs= os.path.join(self.samples_directory, f'iter{i + 1}')
                        if not os.path.exists(path_samples_epochs):
                            os.makedirs(path_samples_epochs)
                        path_origin = os.path.join(path_samples_epochs, name_origin)
                        path_transfer = os.path.join(path_samples_epochs, name_transfer)
                        path_cycle = os.path.join(path_samples_epochs, name_cycle)
                        print(f'[save]:{path_origin},{path_transfer},{path_cycle}')
                        write_pianoroll_save_midis(npy.reshape(1, npy.shape[1], npy.shape[2], npy.shape[0]), f'{path_origin}.mid')
                        np.save(path_origin, npy.reshape(1, npy.shape[0], npy.shape[1], npy.shape[2]))
                        write_pianoroll_save_midis(npy_transfer_binary, f'{path_transfer}.mid')
                        np.save(path_transfer, npy_transfer_binary.reshape(npy_transfer_binary.shape[0], npy_transfer_binary.shape[3], npy_transfer_binary.shape[1], npy_transfer_binary.shape[2]))
                        write_pianoroll_save_midis(npy_cycle_binary, f'{path_cycle}.mid')
                        np.save(path_cycle, npy_cycle_binary.reshape(npy_cycle_binary.shape[0], npy_cycle_binary.shape[3], npy_cycle_binary.shape[1], npy_cycle_binary.shape[2]))

            if (i + 1) % self.model_freq == 0:
                generator_path = os.path.join(self.models_directory, f'{i + 1}-G.ckpt')
                discriminator_path = os.path.join(self.models_directory, f'{i + 1}-D.ckpt')
                classifier_path = os.path.join(self.models_directory, f'{i + 1}-C.ckpt')
                torch.save(self.G.state_dict(), generator_path)
                torch.save(self.D.state_dict(), discriminator_path)
                torch.save(self.C.state_dict(), classifier_path)
                print(f'Models saved into {self.models_directory}.')

            if (i + 1) % self.lr_update_freq == 0 and (i + 1) > (self.epochs - self.lr_decay_epochs):
                generator_lr -= (self.generator_lr / float(self.lr_decay_epochs))
                discriminator_lr -= (self.discriminator_lr / float(self.lr_decay_epochs))
                classifier_lr -= (self.classifier_lr / float(self.lr_decay_epochs))
                for p in self.generator_optimizer.param_groups:
                    p['lr'] = generator_lr
                for p in self.discriminator_optimizer.param_groups:
                    p['lr'] = discriminator_lr
                for p in self.classifier_optimizer.param_groups:
                    p['lr'] = classifier_lr
                print(f'Learning rates decayed. Generator: {generator_lr}, Discriminator: {discriminator_lr}.')

    def test(self):
        print(f'Loading models from {self.test_epochs} epochs')
        generator_path = os.path.join(self.models_directory, f'{self.test_epochs}-G.ckpt')
        discriminator_path = os.path.join(self.models_directory, f'{self.test_epochs}-D.ckpt')
        classifier_path = os.path.join(self.models_directory, f'{self.test_epochs}-C.ckpt')
        self.G.load_state_dict(torch.load(generator_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(discriminator_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(classifier_path, map_location=lambda storage, loc: storage))

        if self.source_style:
            source_style = self.source_style
        else:
            source_style = random.choice(styles)
        files = os.path.join(self.test_directory, source_style)
        files = librosa.util.find_files(files, ext='npy')
        npy_files = {}
        for f in files:
            filename = os.path.basename(f)
            file = np.load(f) * 1.
            if not npy_files.__contains__(filename):
                npy_files[filename] = {}
            npy_files[filename] = file
        target_styles = self.target_style
        for style in target_styles:
            assert style in styles
            label_style = self.styles_encoder.transform([style])[0]
            label_style = np.asarray([label_style])
            with torch.no_grad():
                for filename, npy in npy_files.items():
                    filename = filename.split('.')[0]
                    npy_mod = torch.FloatTensor(npy).to(self.device)
                    npy_mod = npy_mod.view(1, npy_mod.size(0), npy_mod.size(1), npy_mod.size(2))
                    l_style = torch.FloatTensor(label_style)
                    npy_mod = npy_mod.to(self.device)
                    l_style = l_style.to(self.device)
                    npy_transfer = self.G(npy_mod, l_style).cpu().numpy()
                    track_is_max = np.equal(npy_transfer, np.amax(npy_transfer, axis=-1, keepdims=True))
                    track_pass_threshold = (npy_transfer > 0.5)
                    npy_transfer_binary = np.logical_and(track_is_max, track_pass_threshold)
                    npy_transfer_binary = npy_transfer_binary.reshape(-1, npy_transfer_binary.shape[2], npy_transfer_binary.shape[3], npy_transfer_binary.shape[1])
                    name_origin = f'{source_style}-{style}_iter{self.test_epochs}_{filename}_origin'
                    name_transfer = f'{source_style}-{style}_iter{self.test_epochs}_{filename}_transfer'
                    path = os.path.join(self.results_directory, f'iter{self.test_epochs}')
                    path_origin = os.path.join(path, name_origin)
                    path_transfer = os.path.join(path, name_transfer)
                    print(f'saved: {name_origin}, {name_transfer}')
                    write_pianoroll_save_midis(npy.reshape(1, npy.shape[1], npy.shape[2], npy.shape[0]), f'{path_origin}.mid')
                    np.save(path_origin, npy.reshape(1, npy.shape[0], npy.shape[1], npy.shape[2]))
                    write_pianoroll_save_midis(npy_transfer_binary, f'{path_transfer}.mid')
                    np.save(path_transfer, npy_transfer_binary.reshape(npy_transfer_binary.shape[0], npy_transfer_binary.shape[3], npy_transfer_binary.shape[1], npy_transfer_binary.shape[2]))

    def classify(self):
        print("Classify files from " + str(self.classify_directory))
        classifier_path = os.path.join(self.models_directory, f'{self.classifier_epochs}-C.ckpt')
        self.C.load_state_dict(torch.load(classifier_path, map_location=lambda storage, loc: storage))
        files = os.path.join(self.classify_directory)
        files = librosa.util.find_files(files, ext='npy')
        npy_files = {}
        for f in files:
            filename = os.path.basename(f)
            file = np.load(f) * 1.
            if not npy_files.__contains__(filename):
                npy_files[filename] = {}
            npy_files[filename] = file
        with torch.no_grad():
            for filename, npy in npy_files.items():
                npy_mod = torch.FloatTensor(npy).to(self.device)
                try:
                    file_class = self.C(npy_mod)
                    file_class -= file_class.min(1, keepdim=True)[0]
                    file_class /= file_class.max(1, keepdim=True)[0]
                    file_class_style = self.styles_encoder.inverse_transform(file_class.cpu().numpy())
                    print(file_class_style)
                    print(f'File {filename} is classified to {file_class_style[0]} style')
                except:
                    npy_mod = npy_mod.view(1, npy_mod.size(0), npy_mod.size(1), npy_mod.size(2))
                    file_class = self.C(npy_mod)
                    file_class -= file_class.min(1, keepdim=True)[0]
                    file_class /= file_class.max(1, keepdim=True)[0]
                    file_class_style = self.styles_encoder.inverse_transform(file_class.cpu().numpy())
                    print(f'File {filename} is classified to {file_class_style[0]} style')


    def classifier_logs(self):
        print("Classify files from " + str(self.classify_directory))
        classifier_path = os.path.join(self.models_directory, f'{self.classifier_epochs}-C.ckpt')
        self.C.load_state_dict(torch.load(classifier_path, map_location=lambda storage, loc: storage))
        files = os.path.join(self.classify_directory)
        files = librosa.util.find_files(files, ext='npy')
        nr_files = len(files)
        classfied_correctly = 0
        npy_files = {}
        for f in files:
            filename = os.path.basename(f)
            file = np.load(f) * 1.
            if not npy_files.__contains__(filename):
                npy_files[filename] = {}
            npy_files[filename] = file
        with torch.no_grad():
            for filename, npy in npy_files.items():
                npy_mod = torch.FloatTensor(npy).to(self.device)
                npy_mod = npy_mod.view(1, npy_mod.size(0), npy_mod.size(1), npy_mod.size(2))
                gaussian_noise = self.sigma_d * torch.randn(npy_mod.size())
                gaussian_noise = gaussian_noise.to(self.device)
                file_class = self.C(npy_mod + gaussian_noise)
                file_class -= file_class.min(1, keepdim=True)[0]
                file_class /= file_class.max(1, keepdim=True)[0]
                file_class_style = self.styles_encoder.inverse_transform(file_class)
                filename_split = filename.split("_")
                if file_class_style[0] == filename_split[0]:
                    classfied_correctly += 1
                if file_class_style[0] == "RnB" and filename_split[0] == "rock":
                    classfied_correctly += 1
                if file_class_style[0] == "rock" and filename_split[0] == "RnB":
                    classfied_correctly += 1
        # print(classfied_correctly)
        percentage = classfied_correctly*100/nr_files
        print(f'Percentage of the files classified correctly: {percentage} %')