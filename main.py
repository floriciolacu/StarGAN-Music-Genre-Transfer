import os


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    logs_directory = "stargan_songs/logs";
    models_directory = "stargan_songs/models";
    samples_directory = "stargan_songs/samples";
    results_directory = "stargan_songs/results";

    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)
        
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
