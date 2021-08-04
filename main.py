import os
import argparse
from torch.backends import cudnn
import tensorflow as tf
from torch.utils.data.dataset import Dataset
import librosa
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data.dataloader import DataLoader

class AudioDataset(Dataset):
    def __init__(self, datadir:str):
        super(AudioDataset, self).__init__()
        self.datadir = datadir
        self.files = librosa.util.find_files(datadir, ext='npy')
        self.encoder = LabelBinarizer().fit(styles)

    def __getitem__(self, idx):
        p = self.files[idx]
        filename = os.path.basename(p)
        style = filename.split(sep='_', maxsplit=1)[0]
        label = self.encoder.transform([style])[0]
        mid = np.load(p)*1.
        mid = torch.FloatTensor(mid)
        return mid, torch.tensor(styles.index(style), dtype=torch.long), torch.FloatTensor(label)

    def speaker_encoder(self):
        return self.encoder

    def __len__(self):
        return len(self.files)

def str2bool(v):
    return v.lower() in ('true')

def get_styles(dataset_train: str, styles = []):
    if '_' in dataset_train:
        dt = dataset_train.rsplit('_', maxsplit=1)
        styles.append(dt[1])
        get_styles(dt[0],styles)
    else:
        styles.append(dataset_train.rsplit('/', maxsplit=1)[1])
    return list(reversed(styles))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f')

    # Directories.
    parser.add_argument(
        '--dataset_directory',
        type=str,
        default='data/rock_bossanova_funk_RnB',
    )
    parser.add_argument(
        '--test_directory',
        type=str,
        default='data/test',
    )
    parser.add_argument(
        '--logs_directory',
        type=str,
        default='stargan_songs/logs',
    )
    parser.add_argument(
        '--models_directory',
        type=str,
        default='stargan_songs/models',
    )
    parser.add_argument(
        '--samples_directory',
        type=str,
        default='stargan_songs/samples',
    )
    parser.add_argument(
        '--results_directory',
        type=str,
        default='stargan_songs/results',
    )

    parser.add_argument(
        '--cycle_loss_weight',
        type=float,
        default=10,
        help='weight for cycle loss',
    )
    parser.add_argument(
        '--domclass_loss_weight',
        type=float,
        default=5,
        help='weight for domain classification loss',
    )
    parser.add_argument(
        '--identity_loss_weight',
        type=float,
        default=8,
        help='weight for identity loss',
    )
    parser.add_argument(
        '--sigma_d',
        type=float,
        default=0.1,
        help='sigma of gaussian noise for discriminators',
    )

    # Train
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size',
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=200000,
        help='number of total iterations for training Discriminator',
    )
    parser.add_argument(
        '--iters_decay_lr',
        type=int,
        default=100000,
        help='number of iterations for decaying lr',
    )
    parser.add_argument(
        '--generator_lr',
        type=float,
        default=0.0001,
        help='learning rate for Generator',
    )
    parser.add_argument(
        '--discriminator_lr',
        type=float,
        default=0.0001,
        help='learning rate for Discriminator',
    )
    parser.add_argument(
        '--classifier_lr',
        type=float,
        default=0.0001,
        help='learning rate for Classifier',
    )
    parser.add_argument(
        '--discriminator_updates',
        type=int,
        default=5,
        help='number of Discriminator updates per each Generator update',
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='beta1 for Adam optimizer',
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='beta2 for Adam optimizer',
    )
    parser.add_argument(
        '--resume_iters',
        type=int,
        default=None,
        help='resume training from this step',
    )

    # Test
    parser.add_argument(
        '--test_iters',
        type=int,
        default=200000,
        help='test model from this step',
    )
    parser.add_argument(
        '--source_style',
        type=str,
        default=None,
        help='test model source style',
    )
    parser.add_argument(
        '--target_style',
        type=str,
        default="['rock', 'bossanova']",
        help='string list of target styles eg."[a,b]"',
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--type',
        type=str,
        default='train',
        choices=['train', 'test', 'classify'],
    )
    # parser.add_argument(
    #     '--use_tensorboard',
    #     type=str2bool,
    #     default=True,
    # )

    # Steps
    parser.add_argument(
        '--log_freq',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--sample_freq',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--model_freq',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--lr_update_freq',
        type=int,
        default=100000,
    )

    args = parser.parse_args()
    print(args)

    cudnn.benchmark = True

    if not os.path.exists(args.logs_directory):
        os.makedirs(args.logs_directory)
    if not os.path.exists(args.models_directory):
        os.makedirs(args.models_directory)
    if not os.path.exists(args.samples_directory):
        os.makedirs(args.samples_directory)
    if not os.path.exists(args.results_directory):
        os.makedirs(args.results_directory)

    styles = get_styles('./data/rock_bossanova_funk_RnB')
    dataset = AudioDataset(args.dataset_directory)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Solver for training and testing StarGAN.
    # solver = Solver(dloader, config)

    if args.type == "train":
        print("train")
        # model = CycleGAN(args)
        # model.train(args) if args.phase == "train" else model.test(args)

    if args.type == "test":
        print("test")
        # classifier = Classifier(args)

    if args.type == "classify":
        print("classify")
        # code

