import os
import torchaudio
from datasets import Dataset, Audio
from tqdm import tqdm
import argparse
from huggingface_hub import snapshot_download


def fisher_dataset_for_speaker_diarization(fpath, nb_files): 

    txt_files = list()
    txt_filenames = list()
    sph_files = list()
    sph_filenames = list()

    # get the audio and transcription directories -> no info about fisher directory structure required
    for (dirpath, dirnames, filenames) in os.walk(fpath):
        # get the audio (.sph) and transcription (.txt) file paths
        sph_filenames += [file for file in filenames if file.endswith(".sph")]
        sph_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".sph")]
        txt_filenames += [file for file in filenames if file.endswith(".txt")]
        txt_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".txt")]

    # now iterate over all transcriptions
    for file_idx, file in tqdm(enumerate(txt_files[:nb_files])):
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

            samples = segment[0] + segment[1]

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
                "audio": {"path": file, "array": samples, "sampling_rate": sampling_rate},
                "timestamps_start": timestamps_start,
                "timestamps_end": timestamps_end,
                "speakers": speakers,
                "transcripts": transcripts, 
            }

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--download', default = False)
    parser.add_argument('--local_fisher_dir', default= "/raid/kamilakesbi/fisher")

    parser.add_argument('--preprocess', default=False)
    parser.add_argument('--nb_files', default=8000)
    parser.add_argument('--preprocess_cache_dir', default='/raid/kamilakesbi/')
    parser.add_argument('--hub_folder', default='kamilakesbi/fisher')
    
    args = parser.parse_args()

    if args.download is True: 

        snapshot_download(repo_id="speech-seq2seq/fisher", repo_type="dataset", local_dir=args.local_fisher_dir)

    if args.preprocess is True: 

        dataset = Dataset.from_generator(
            lambda x : fisher_dataset_for_speaker_diarization(x, fpath= os.path.join(args.local_fisher_dir,'/data'), nb_files=int(args.nb_files)), 
            writer_batch_size=200, 
            cache_dir=args.preprocess_cache_dir, 
        )
        dataset = dataset.cast_column("audio", Audio())
        dataset.push_to_hub(args.hub_folder, private=True)
