from torch.backends import cudnn
import os
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import LabelBinarizer
import librosa
from StarGAN import *


class SongsDataset(Dataset):
    def __init__(self, dataset_directory:str):
        super(SongsDataset, self).__init__()
        self.dataset_directory = dataset_directory
        self.files = librosa.util.find_files(dataset_directory, ext='npy')
        self.style_encoder = LabelBinarizer().fit(styles)

    def __getitem__(self, idx):
        p = self.files[idx]
        filename = os.path.basename(p)
        style = filename.split(sep='_', maxsplit=1)[0]
        label = self.style_encoder.transform([style])[0]
        file = np.load(p)*1.
        file = torch.FloatTensor(file)
        return file, torch.tensor(styles.index(style), dtype=torch.long), torch.FloatTensor(label)

    def __len__(self):
        return len(self.files)


def str2bool(s):
    return s.lower() in ('true')


def get_styles(dataset_train: str, styles = []):
    if '_' in dataset_train:
        dataset_train = dataset_train.rsplit('_', maxsplit=1)
        styles.append(dataset_train[1])
        get_styles(dataset_train[0],styles)
    else:
        styles.append(dataset_train.rsplit('/', maxsplit=1)[1])
    return list(reversed(styles))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('-f')

    # Directories
    parser.add_argument(
        '--dataset_directory',
        type=str,
        default='data/RnB_bossanova_funk_rock',
        help="path of the dataset directory for train",
    )
    parser.add_argument(
        '--test_directory',
        type=str,
        default='data/test1',
        help="path of the dataset directory for test",
    )
    parser.add_argument(
        '--classify_directory',
        type=str,
        default='data/test1',
        help="path of the directory of the files to classify",
    )
    parser.add_argument(
        '--logs_directory',
        type=str,
        default='stargan_songs/logs',
        help="logs are saved here",
    )
    parser.add_argument(
        '--models_directory',
        type=str,
        default='stargan_songs/models',
        help="models are saved here",
    )
    parser.add_argument(
        '--samples_directory',
        type=str,
        default='stargan_songs/samples',
        help="samples are saved here",
    )
    parser.add_argument(
        '--results_directory',
        type=str,
        default='stargan_songs/results',
        help="samples are saved here",
    )
    # Model
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
        help='# songs in batch',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200000,
        help='# of epochs to train Discriminator',
    )
    parser.add_argument(
        '--lr_decay_epochs',
        type=int,
        default=100000,
        help='# of epochs to decaying lr',
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
        '--resume_epochs',
        type=int,
        default=None,
        help='resume train from # of epochs',
    )
    # Test and classifier
    parser.add_argument(
        '--test_epochs',
        type=int,
        default=200000,
        help='test model from # of epochs',
    )
    parser.add_argument(
        '--classifier_epochs',
        type=int,
        default=200000,
        help='classifier model from # of epochs',
    )
    parser.add_argument(
        '--source_style',
        type=str,
        default=None,
        help='source style for testing',
    )
    parser.add_argument(
        '--target_style',
        type=str,
        default="['rock', 'bossanova']",
        help='list of target styles for testing eg."[a,b]"',
    )
    # Others
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='# of workers that will simultaneously retrieve data',
    )
    parser.add_argument(
        '--type',
        type=str,
        default='train',
        choices=['train', 'test', 'classify'],
        help='train, test or classify',
    )
    parser.add_argument(
        '--use_tensorboard',
        type=str2bool,
        default=True,
        help='tensorboard to be used',
    )
    # Frequencies
    parser.add_argument(
        '--print_freq',
        type=int,
        default=10,
        help="print info every print_freq epochs"
    )
    parser.add_argument(
        '--sample_freq',
        type=int,
        default=2000,
        help="save samples every sample_freq epochs"
    )
    parser.add_argument(
        '--model_freq',
        type=int,
        default=10000,
        help="save model every model_freq epochs"
    )
    parser.add_argument(
        '--lr_update_freq',
        type=int,
        default=100000,
        help="update learning rate every lr_update_freq epochs"
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

    styles = get_styles('./data/RnB_bossanova_funk_rock')
    songs_dataset = SongsDataset(args.dataset_directory)
    dataset_loader = DataLoader(songs_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.type == "train":
        model = StarGAN(dataset_loader, args)
        model.train()

    if args.type == "test":
        model = StarGAN(dataset_loader, args)
        model.test()

    if args.type == "classify":
        model = StarGAN(dataset_loader, args)
        model.classify()
        # model.classifier_logs()

