import os
from datetime import datetime
import tensorflow as tf
import pretty_midi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Discriminator, Classifier, Generator
import random
from sklearn.preprocessing import LabelBinarizer
import librosa
import ast


def get_styles(dataset_train: str, styles = []):
    if '_' in dataset_train:
        dt = dataset_train.rsplit('_', maxsplit=1)
        styles.append(dt[1])
        get_styles(dt[0],styles)
    else:
        styles.append(dataset_train.rsplit('/', maxsplit=1)[1])
    return list(reversed(styles))


styles = get_styles('./data/rock_bossanova_funk_RnB')


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
            # set_piano_roll_to_instrument(images_with_pause_list[idx], instrument, 100, tempo, 4)
            # def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16):
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
    def __init__(self, loader, args):
        self.args = args
        self.loader = loader

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
        self.discriminator_optimizer = torch.optim.Adam(self.D.parameters(), self.discriminator_lr,
                                                        [self.beta1, self.beta2])
        self.classifier_optimizer = torch.optim.Adam(self.C.parameters(), self.classifier_lr, [self.beta1, self.beta2])

        networks = [self.G, self.D, self.C]
        for n in networks:
            num_params = 0
            for p in n.parameters():
                num_params += p.numel()
            print(n)
            print("The number of parameters: {}".format(num_params))

        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)
        if self.use_tensorboard:
            self.writer = tf.summary.create_file_writer(self.logs_directory)

    def train(self):
        generator_lr = self.generator_lr
        discriminator_lr = self.discriminator_lr
        classifier_lr = self.classifier_lr

        start_iters = 0
        if self.resume_epochs:
            pass

        # norm = Normalizer()
        data_iter = iter(self.loader)

        print('Start training......')
        start_time = datetime.now()

        for i in range(start_iters, self.epochs):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            # Fetch real images and labels.
            try:
                x_real, style_idx_org, label_org = next(data_iter)
            except:
                data_iter = iter(self.loader)
                x_real, style_idx_org, label_org = next(data_iter)

                # generate gaussian noise for robustness improvement

            gaussian_noise = self.sigma_d * torch.randn(x_real.size())

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            style_idx_trg = style_idx_org[rand_idx]

            x_real = x_real.to(self.device)  # Input images.
            label_org = label_org.to(self.device)  # Original domain one-hot labels.
            label_trg = label_trg.to(self.device)  # Target domain one-hot labels.
            style_idx_org = style_idx_org.to(self.device)  # Original domain labels
            style_idx_trg = style_idx_trg.to(self.device)  # Target domain labels
            gaussian_noise = gaussian_noise.to(self.device)  # gaussian noise for discriminators
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Compute loss with real audio frame.
            CELoss = nn.CrossEntropyLoss()
            cls_real = self.C(x_real)
            # print(x_real.shape)
            cls_loss_real = CELoss(input=cls_real, target=style_idx_org)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()

            cls_loss_real.backward()
            self.classifier_optimizer.step()
            # Logging.
            loss = {}
            loss['C/C_loss'] = cls_loss_real.item()

            # print(x_real.shape)
            out_r = self.D(x_real + gaussian_noise, label_org)
            # Compute loss with fake audio frame.
            x_fake = self.G(x_real, label_trg)
            out_f = self.D(x_fake + gaussian_noise, label_trg)
            d_loss_t = F.mse_loss(input=out_f, target=torch.zeros_like(out_f, dtype=torch.float)) + \
                       F.mse_loss(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))

            out_cls = self.C(x_fake)
            d_loss_cls = CELoss(input=out_cls, target=style_idx_trg)

            # Compute loss for gradient penalty.
            # alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            # out_src = self.D(x_hat, label_trg)
            # d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_loss_t + self.domclass_loss_weight * d_loss_cls
            # \+ 5*d_loss_gp

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()

            d_loss.backward()
            self.discriminator_optimizer.step()

            # loss['D/d_loss_t'] = d_loss_t.item()
            # loss['D/loss_cls'] = d_loss_cls.item()
            # loss['D/D_gp'] = d_loss_gp.item()
            loss['D/D_loss'] = d_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            if (i + 1) % self.discriminator_updates == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, label_trg)
                g_out_src = self.D(x_fake + gaussian_noise, label_trg)
                g_loss_fake = F.mse_loss(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))

                out_cls = self.C(x_real)
                g_loss_cls = CELoss(input=out_cls, target=style_idx_org)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, label_org)
                g_loss_rec = F.l1_loss(x_reconst, x_real)

                # Original-to-Original domain(identity).
                x_fake_iden = self.G(x_real, label_org)
                id_loss = F.l1_loss(x_fake_iden, x_real)

                # Backward and optimize.
                g_loss = g_loss_fake + self.cycle_loss_weight * g_loss_rec + \
                         self.domclass_loss_weight * g_loss_cls + self.identity_loss_weight * id_loss

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                self.classifier_optimizer.zero_grad()

                g_loss.backward()
                self.generator_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_id'] = id_loss.item()
                loss['G/g_loss'] = g_loss.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i + 1) % self.log_freq == 0:
                et = datetime.now() - start_time
                et = str(et)[:-7]
                log = "Elapsed [{}], Epochs [{}/{}]".format(et, i + 1, self.epochs)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        # self.logger.scalar_summary(tag, value, i + 1)
                        summary = tf.summary.scalar(tag, value, step=i + 1)
                        self.writer.flush()

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_freq == 0:
                with torch.no_grad():
                    # d, style = TestSet(self.test_directory).test_data()
                    if self.source_style:
                        r_s = self.source_style
                    else:
                        r_s = random.choice(styles)
                    p = os.path.join(self.test_directory, r_s)
                    npyfiles = librosa.util.find_files(p, ext='npy')
                    res = {}
                    for f in npyfiles:
                        filename = os.path.basename(f)
                        print(filename)
                        mid = np.load(f) * 1.
                        if not res.__contains__(filename):
                            res[filename] = {}
                        res[filename] = mid
                    d, style = res, r_s
                    label_o = self.styles_encoder.transform([style])[0]
                    label_o = np.asarray([label_o])
                    target = random.choice([x for x in styles if x != style])
                    label_t = self.styles_encoder.transform([target])[0]
                    label_t = np.asarray([label_t])
                    for filename, content in d.items():
                        filename = filename.split('.')[0]
                        one_seg = torch.FloatTensor(content).to(self.device)
                        one_seg = one_seg.view(1, one_seg.size(0), one_seg.size(1), one_seg.size(2))
                        l_t = torch.FloatTensor(label_t)
                        one_seg = one_seg.to(self.device)
                        l_t = l_t.to(self.device)
                        one_set_transfer = self.G(one_seg, l_t)
                        l_o = torch.FloatTensor(label_o)
                        l_o = l_o.to(self.device)
                        one_set_cycle = self.G(one_set_transfer.to(self.device), l_o).data.cpu().numpy()
                        one_set_transfer = one_set_transfer.data.cpu().numpy()
                        # one_set_transfer_binary = to_binary(one_set_transfer, 0.5)
                        # one_set_cycle_binary = to_binary(one_set_cycle, 0.5)
                        track_is_max = np.equal(one_set_transfer, np.amax(one_set_transfer, axis=-1, keepdims=True))
                        track_pass_threshold = (one_set_transfer > 0.5)
                        one_set_transfer_binary = np.logical_and(track_is_max, track_pass_threshold)

                        track_is_max_cycle = np.equal(one_set_cycle, np.amax(one_set_cycle, axis=-1, keepdims=True))
                        track_pass_threshold_cycle = (one_set_cycle > 0.5)
                        one_set_cycle_binary = np.logical_and(track_is_max_cycle, track_pass_threshold_cycle)

                        one_set_transfer_binary = one_set_transfer_binary.reshape(-1, one_set_transfer_binary.shape[2],
                                                                                  one_set_transfer_binary.shape[3],
                                                                                  one_set_transfer_binary.shape[1])
                        one_set_cycle_binary = one_set_cycle_binary.reshape(-1, one_set_cycle_binary.shape[2],
                                                                            one_set_cycle_binary.shape[3],
                                                                            one_set_cycle_binary.shape[1])
                        # print(one_set_transfer_binary.shape, one_set_cycle_binary.shape)
                        name_origin = f'{style}-{target}_iter{i + 1}_{filename}_origin'
                        name_transfer = f'{style}-{target}_iter{i + 1}_{filename}_transfer'
                        name_cycle = f'{style}-{target}_iter{i + 1}_{filename}_cycle'
                        path_samples_per_iter = os.path.join(self.samples_directory, f'iter{i + 1}')
                        if not os.path.exists(path_samples_per_iter):
                            os.makedirs(path_samples_per_iter)
                        path_origin = os.path.join(path_samples_per_iter, name_origin)
                        path_transfer = os.path.join(path_samples_per_iter, name_transfer)
                        path_cycle = os.path.join(path_samples_per_iter, name_cycle)
                        print(f'[save]:{path_origin},{path_transfer},{path_cycle}')
                        write_pianoroll_save_midis(content.reshape(1, content.shape[1], content.shape[2], content.shape[0]),
                                   '{}.mid'.format(path_origin))
                        write_pianoroll_save_midis(one_set_transfer_binary, '{}.mid'.format(path_transfer))
                        write_pianoroll_save_midis(one_set_cycle_binary, '{}.mid'.format(path_cycle))

            if (i + 1) % self.model_freq == 0:
                G_path = os.path.join(self.models_directory, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.models_directory, '{}-D.ckpt'.format(i + 1))
                C_path = os.path.join(self.models_directory, '{}-C.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.models_directory))

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
                print('Decayed learning rates, generator_lr: {}, discriminator_lr: {}.'.format(generator_lr, discriminator_lr))

    def test(self):
        print('Loading the trained models from step {}...'.format(self.test_epochs))
        G_path = os.path.join(self.models_directory, '{}-G.ckpt'.format(self.test_epochs))
        D_path = os.path.join(self.models_directory, '{}-D.ckpt'.format(self.test_epochs))
        C_path = os.path.join(self.models_directory, '{}-C.ckpt'.format(self.test_epochs))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        if self.source_style:
            r_s = self.source_style
        else:
            r_s = random.choice(styles)
        p = os.path.join(self.test_directory, r_s)
        npyfiles = librosa.util.find_files(p, ext='npy')
        res = {}
        for f in npyfiles:
            filename = os.path.basename(f)
            mid = np.load(f) * 1.
            if not res.__contains__(filename):
                res[filename] = {}
            res[filename] = mid
        d, style = res, r_s
        targets = self.target_style
        for target in targets:
            assert target in styles
            label_t = self.styles_encoder.transform([target])[0]
            label_t = np.asarray([label_t])
            with torch.no_grad():
                for filename, content in d.items():
                    filename = filename.split('.')[0]
                    one_seg = torch.FloatTensor(content).to(self.device)
                    one_seg = one_seg.view(1, one_seg.size(0), one_seg.size(1), one_seg.size(2))
                    l_t = torch.FloatTensor(label_t)
                    one_seg = one_seg.to(self.device)
                    l_t = l_t.to(self.device)
                    one_set_transfer = self.G(one_seg, l_t).cpu().numpy()
                    track_is_max = np.equal(one_set_transfer, np.amax(one_set_transfer, axis=-1, keepdims=True))
                    track_pass_threshold = (one_set_transfer > 0.5)
                    one_set_transfer_binary = np.logical_and(track_is_max, track_pass_threshold)
                    one_set_transfer_binary = one_set_transfer_binary.reshape(-1, one_set_transfer_binary.shape[2],
                                                                              one_set_transfer_binary.shape[3],
                                                                              one_set_transfer_binary.shape[1])
                    name_origin = f'{style}-{target}_iter200000_{filename}_origin'
                    name_transfer = f'{style}-{target}_iter200000_{filename}_transfer'
                    path = os.path.join(self.results_directory, f'iter200000')
                    path_origin = os.path.join(path, name_origin)
                    path_transfer = os.path.join(path, name_transfer)
                    print(f'[save]:{path_origin},{path_transfer}')
                    write_pianoroll_save_midis(content.reshape(1, content.shape[1], content.shape[2], content.shape[0]),
                               '{}.mid'.format(path_origin))
                    write_pianoroll_save_midis(one_set_transfer_binary, '{}.mid'.format(path_transfer))

    def classify(self):
        print("Classify files from directory " + str(self.classify_directory))
        C_path = os.path.join(self.models_directory, '{}-C.ckpt'.format(self.classifier_epochs))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
        p = os.path.join(self.classify_directory)
        npyfiles = librosa.util.find_files(p, ext='npy')
        res = {}
        for f in npyfiles:
            filename = os.path.basename(f)
            mid = np.load(f) * 1.
            if not res.__contains__(filename):
                res[filename] = {}
            res[filename] = mid
        with torch.no_grad():
            for filename, content in res.items():
                one_seg = torch.FloatTensor(content).to(self.device)
                one_seg = one_seg.view(1, one_seg.size(0), one_seg.size(1), one_seg.size(2))
                test_class = self.C(one_seg)
                test_class -= test_class.min(1, keepdim=True)[0]
                test_class /= test_class.max(1, keepdim=True)[0]
                test_class_style = self.styles_encoder.inverse_transform(test_class)
                print(filename + " has style class " + test_class_style[0])



