import os
import torchaudio
from tqdm import tqdm
import argparse
from datasets import Dataset, Audio, DatasetDict
import torchaudio.transforms as T


def fisher_dataset_for_speaker_diarization(fpath="/data/fisher/data", split = 'train'): 

    txt_files = list()
    txt_filenames = list()
    sph_files = list()
    sph_filenames = list()

    with open('preprocessing/fisher_eval.txt', 'r') as file:
        test_files = [line.strip() for line in file.readlines()]

    # get the audio and transcription directories -> no info about fisher directory structure required
    for (dirpath, dirnames, filenames) in os.walk(fpath):
        # get the audio (.sph) and transcription (.txt) file paths
        sph_filenames += [file for file in filenames if file.endswith(".sph")]
        sph_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".sph")]
        txt_filenames += [file for file in filenames if file.endswith(".txt")]
        txt_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".txt")]

    # now iterate over all transcriptions
    for file_idx, file in tqdm(enumerate(txt_files)):
        # get the transcription filename without path (matches the corresponding ".sph" name)
        txt_filename = txt_filenames[file_idx].rstrip(".txt")
        
        timestamps_start = []
        timestamps_end = []
        speakers = []
        transcripts = []

        # ignore non-transcription files
        if "readme" not in txt_filename and "doc" not in txt_filename:
            # get the corresponding audio file and load
            sph_idx = sph_filenames.index(txt_filename+".sph")

            segment, sampling_rate = torchaudio.load(sph_files[sph_idx], format="sph")
        
            # transform = T.Resample(8000, 16000)
            # segment = transform(segment)
            audio_path = sph_files[sph_idx].split('.')[0] + '.wav'
            
            filename = audio_path.split('/')[-1].split('.')[0]

            if (filename in test_files and split=='test') or (filename not in test_files and split=='train'): 

                # torchaudio.save(uri = audio_path, src=segment, format = 'wav', sample_rate=16000)
                # segment, sr = torchaudio.load(uri = audio_path, format='wav')

                with open(file) as f:
                    for line in f:
                        # only parse non-empty lines
                        if line.strip():
                            line = line.strip()
                            # remove double spaces between columns in the transcription
                            line = " ".join(line.split())
                            if line.startswith("#"):
                                continue
                            # split the line as before, this time according to the new column headings
                            start, end, speaker, transcript = line.split(" ", 3)

                            timestamps_start.append(float(start))
                            timestamps_end.append(float(end))
                            speakers.append(speaker[0])
                            transcripts.append(transcript)
                yield {
                    "audio": {"path": audio_path, "array": segment[0] + segment[1], "sampling_rate": sampling_rate},
                    "timestamps_start": timestamps_start,
                    "timestamps_end": timestamps_end,
                    "speakers": speakers,
                    "transcripts": transcripts, 
                    "filename": filename, 
                }


if __name__ == "__main__":

    preprocess_cache_dir = '/data/fisher'
    hub_folder = 'kamilakesbi/fisher'

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--download', default = False)
    parser.add_argument('--local_fisher_dir', default= "/data/fisher/data")
    parser.add_argument('--preprocess', default=True)
    parser.add_argument('--preprocess_cache_dir', default='/data/fisher')
    parser.add_argument('--hub_folder', default='kamilakesbi/fisher')
    
    args = parser.parse_args()

    # if args.download: 
    #     snapshot_download(repo_id="speech-seq2seq/fisher", repo_type="dataset", local_dir=args.local_fisher_dir)

    # if args.preprocess: 
    
    dataset = DatasetDict(
        {
            'train': [], 
            'test': [], 
        }
    )

    dataset['train'] = Dataset.from_generator(
        fisher_dataset_for_speaker_diarization,
        gen_kwargs={'fpath': args.local_fisher_dir, 'split': 'train'},
        writer_batch_size=200,
        cache_dir=args.preprocess_cache_dir, 
    )

    dataset['test'] = Dataset.from_generator(
        fisher_dataset_for_speaker_diarization,
        gen_kwargs={'fpath': args.local_fisher_dir, 'split': 'test'},
        writer_batch_size=200,
        cache_dir=args.preprocess_cache_dir, 
    )

    dataset.push_to_hub(hub_folder, private=True)
