import json
import pathlib
import pickle
import random
from collections import namedtuple
from pathlib import Path

import PIL.Image
import PIL.Image
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

tokenizer = None

SpecialTokens = namedtuple('SpecialTokens', ['eos', 'pad', 'sos', 'unk'])
special_tokens = SpecialTokens('<eos>', '<pad>', '<sos>', '<unk>')

# Number of files to keep in the debug mode
NB_DEBUG = 50


def init_vocabulary(transcriptions):
    global tokenizer
    tokenizer = LabelEncoder()
    tokens = list(special_tokens) + [c for t in transcriptions for c in t]
    tokenizer.fit(tokens)


def get_token_id(token):
    return tokenizer.transform([token])[0]


def caption2tensor(capt):
    capt = [c if c in tokenizer.classes_ else special_tokens.unk for c in capt]
    capt = [special_tokens.sos] + capt + [special_tokens.eos]
    return torch.Tensor(tokenizer.transform(capt))


class Flickr8KData(torch.utils.data.Dataset):
    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [sd[2] for sd in dataset.split_data]
        init_vocabulary(transcriptions)

    def __init__(self, root, feature_fname, meta_fname, split='train', language='en',
                 downsampling_factor=None):
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        self.language = language
        if language == 'en':
            self.text_key = 'raw'
        elif language == 'jp':
            self.text_key = 'raw_jp'
        else:
            raise ValueError('Language {} not supported.'.format(language))
        self.root = root
        self.split = split
        self.language = language
        root_path = pathlib.Path(root)
        # Loading label encoder
        module_path = pathlib.Path(__file__).parent
        with open(module_path / 'label_encoders.pkl', 'rb') as f:
            global tokenizer
            tokenizer = pickle.load(f)[language]
        # Loading metadata
        with open(root_path / meta_fname) as fmeta:
            metadata = json.load(fmeta)['images']
        # Loading mapping from image id to list of caption id
        self.image_captions = {}
        with open(root_path / 'flickr_audio' / 'wav2capt.txt') as fwav2capt:
            for line in fwav2capt:
                audio_id, image_id, text_id = line.split()
                text_id = int(text_id[1:])
                self.image_captions[image_id] = self.image_captions.get(image_id, []) + [(text_id, audio_id)]

        # Creating image, caption pairs
        self.split_data = []
        for image in metadata:
            if image['split'] == self.split:
                fname = image['filename']
                for text_id, audio_id in self.image_captions[fname]:
                    # In the reduced dataset containing only sentences with
                    # translations, removed sentences are replaced by 'None' to
                    # keep the index of the sentence fixed, so that we can
                    # still retrieve them based on text_id.
                    # TODO: find a nicer way to handle this
                    if image['sentences'][text_id] is not None:
                        if self.text_key in image['sentences'][text_id]:
                            self.split_data.append((
                                fname,
                                audio_id,
                                image['sentences'][text_id][self.text_key]))

        # Downsampling
        if downsampling_factor is not None:
            num_examples = int(len(self.split_data) // downsampling_factor)
            self.split_data = random.sample(self.split_data, num_examples)

        # image and audio feature data
        image = torch.load(root_path / 'resnet_features.pt')
        self.image = dict(zip(image['filenames'], image['features']))
        audio = torch.load(root_path / feature_fname)
        self.audio = dict(zip(audio['filenames'], audio['features']))

    def __getitem__(self, index):
        sd = self.split_data[index]
        image = self.image[sd[0]]
        audio = self.audio[sd[1]]
        text = caption2tensor(sd[2])
        return dict(image_id=sd[0],
                    audio_id=sd[1],
                    image=image,
                    text=text,
                    audio=audio,
                    gloss=sd[2])

    def __len__(self):
        return len(self.split_data)

    def get_config(self):
        return dict(feature_fname=self.feature_fname,
                    label_encoder=self.get_label_encoder(),
                    language=self.language)

    def evaluation(self):
        """Returns image features, audio features, caption features, and a
        boolean array specifying whether a caption goes with an image."""
        audio = []
        text = []
        image = []
        matches = []
        image2idx = {}
        for sd in self.split_data:
            # Add image
            if sd[0] in image2idx:
                image_idx = image2idx[sd[0]]
            else:
                image_idx = len(image)
                image2idx[sd[0]] = image_idx
                image.append(self.image[sd[0]])
            # Add audio and text
            audio.append(self.audio[sd[1]])
            text.append(sd[2])
            matches.append((len(audio) - 1, image_idx))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i, j in matches:
            correct[i, j] = True
        return dict(image=image, audio=audio, text=text, correct=correct)

    def is_slt(self):
        return self.language != 'en'

    def split_sentences(self, sentences):
        if self.language == 'jp':
            return sentences
        else:
            return [s.split() for s in sentences]


class LibriSpeechData(torch.utils.data.Dataset):
    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [m['trn'] for m in dataset.metadata]
        init_vocabulary(transcriptions)

    def __init__(self, root, feature_fname, meta_fname, split='train',
                 downsampling_factor=None):
        # 'val' set in flickr8k corresponds to 'dev' in librispeech
        if split == 'val':
            split = 'dev'
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        root_path = pathlib.Path(root)
        with open(root_path / meta_fname) as fmeta:
            self.metadata = json.load(fmeta)
            self.num_lines = self.metadata[-1]['audio_end']
        if downsampling_factor is not None:
            num_examples = len(self.metadata) // downsampling_factor
            self.metadata = random.sample(self.metadata, num_examples)
        # filter examples based on split
        meta = []
        for ex in self.metadata:
            if ex['split'] == self.split:
                meta.append(ex)
        self.metadata = meta
        # load audio features
        self.audio = np.memmap(root_path / feature_fname, dtype='float64',
                               mode='r', shape=(self.num_lines, 39))

    def __getitem__(self, index):
        sd = self.metadata[index]
        audio = torch.from_numpy(self.audio[sd['audio_start']:sd['audio_end']])
        text = caption2tensor(sd['trn'])
        return dict(audio_id=sd['fileid'], text=text, audio=audio)

    def __len__(self):
        return len(self.metadata)

    def get_config(self):
        return dict(feature_fname=self.feature_fname,
                    label_encoder=self.get_label_encoder())

    def evaluation(self):
        """Returns audio features with corresponding caption"""
        audio = []
        text = []
        for ex in self.metadata:
            text.append(ex['trn'])
            a = torch.from_numpy(self.audio[ex['audio_start']:ex['audio_end']])
            audio.append(a)
        return dict(audio=audio, text=text)


class SpokenCOCOData(torch.utils.data.Dataset):
    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [sd[2] for sd in dataset.split_data]
        init_vocabulary(transcriptions)

    def __init__(self, root, feature_fname, meta_fname, split='train',
                 downsampling_factor=None, debug=False):

        if feature_fname.endswith('.pt'):
            raise ValueError("Training on SpokenCOCO is not possible with .pt files. Please provide --audio_features_fn "
                             "that points to .memmap files.")

        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        self.language = 'en'
        self.root = root
        self.split = split
        root_path = pathlib.Path(root)
        # Loading label encoder
        module_path = pathlib.Path(__file__).parent
        with open(module_path / 'label_encoders.pkl', 'rb') as f:
            global tokenizer
            tokenizer = pickle.load(f)[self.language]

        if split != "train":
            if split == "val":
                meta_fname = meta_fname.replace('train', 'val')
            else:
                raise ValueError("Split == %s not defined for SpokenCOCO" % split)

        # Loading metadata
        with open(root_path / meta_fname) as fmeta:
            metadata = json.load(fmeta)['data']

        image_feature_fname = 'resnet_features.memmap'
        if debug:
            image_feature_fname = image_feature_fname.replace('.memmap', '_debug.memmap')
            feature_fname = feature_fname.replace('.memmap', '_debug.memmap')
            metadata = metadata[0:NB_DEBUG]

        # Loading mapping from image id to list of caption id
        self.image_captions = {}
        # And creating image, caption pairs
        self.split_data = []
        for sample in metadata:
            img_id = sample['image']
            caption_list = []
            for caption in sample['captions']:
                caption_list.append((caption['uttid'], caption['wav']))
                self.split_data.append((img_id, caption['wav'], caption['text']))
            self.image_captions[img_id] = caption_list

        # Downsampling
        if downsampling_factor is not None:
            num_examples = int(len(self.split_data) // downsampling_factor)
            self.split_data = random.sample(self.split_data, num_examples)

        # Load memory map metadata
        mmap_meta_fname = root_path / (image_feature_fname.replace('.memmap', '_memmap_mapping.json'))
        with open(root_path / mmap_meta_fname) as fmeta:
            self.image_mmap_mapping = json.load(fmeta)
            last_key = list(self.image_mmap_mapping)[-1]
            self.nb_images = self.image_mmap_mapping[last_key]['image_idx']

        mmap_meta_fname = root_path / (feature_fname.replace('.memmap', '_memmap_mapping.json'))
        with open(root_path / mmap_meta_fname) as fmeta:
            self.audio_mmap_mapping = json.load(fmeta)
            last_key = list(self.audio_mmap_mapping)[-1]
            self.nb_frames = self.audio_mmap_mapping[last_key]['audio_end']

        # Load memory-map arrays
        self.image = np.memmap(root_path / image_feature_fname, dtype='float32',
                               mode='r', shape=(self.nb_images+1, self.image_mmap_mapping['feature_size']))

        self.audio = np.memmap(root_path / feature_fname, dtype='float64',
                               mode='r', shape=(self.nb_frames, self.audio_mmap_mapping['feature_size']))

    def __getitem__(self, index):
        # Get ids and captions
        sd = self.split_data[index]
        image_id = sd[0]
        audio_id = sd[1]
        caption = sd[2]

        # Get their location in the memory-map array
        image_idx = self.image_mmap_mapping[image_id]['image_idx']
        audio_start = self.audio_mmap_mapping[audio_id]['audio_start']
        audio_end = self.audio_mmap_mapping[audio_id]['audio_end']

        # Retrieve features
        image = torch.from_numpy(self.image[image_idx])
        audio = torch.from_numpy(self.audio[audio_start:audio_end])
        text = caption2tensor(caption)

        return dict(image_id=image_id,
                    audio_id=audio_id,
                    image=image,
                    audio=audio,
                    text=text,
                    gloss=caption)

    def __len__(self):
        return len(self.split_data)

    def get_config(self):
        return dict(feature_fname=self.feature_fname,
                    label_encoder=self.get_label_encoder())

    def evaluation(self):
        """Returns image features, audio features, caption features, and a
        boolean array specifying whether a caption goes with an image."""
        audio = []
        text = []
        image = []
        matches = []
        image2idx = {}
        for sd in self.split_data:
            # Get ids
            image_id = sd[0]
            audio_id = sd[1]
            caption = sd[2]

            # Get their location in the memory-map array
            memmap_image_idx = self.image_mmap_mapping[image_id]['image_idx']
            memmap_audio_start = self.audio_mmap_mapping[audio_id]['audio_start']
            memmap_audio_end = self.audio_mmap_mapping[audio_id]['audio_end']

            # Add image
            if image_id in image2idx:
                image_idx = image2idx[image_id]
            else:
                image_idx = len(image)
                image2idx[image_id] = image_idx
                image.append(torch.from_numpy(self.image[memmap_image_idx]))

            # Add audio and text
            audio.append(torch.from_numpy(self.audio[memmap_audio_start:memmap_audio_end]))
            text.append(caption)
            matches.append((len(audio) - 1, image_idx))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i, j in matches:
            correct[i, j] = True

        return dict(image=image, audio=audio, text=text, correct=correct)


# Util functions to create batch
def batch_audio(audios, max_frames=2048):
    """Merge audio captions. Truncate to max_frames. Pad with 0s."""
    permute = False
    if len(audios[0].shape) == 1:
        # Then we're working from the waveform
        audio_lengths = [len(cap[:max_frames]) for cap in audios]
        # nb audio * max_len
        audio = torch.zeros(len(audios), max(audio_lengths))
    else:
        # Then we're working from MFCCs
        audio_lengths = [len(cap[:max_frames, :]) for cap in audios]
        # nb audio * max_len * n_features
        audio = torch.zeros(len(audios), max(audio_lengths), audios[0].size(1))
        permute = True

    for i, cap in enumerate(audios):
        end = audio_lengths[i]
        audio[i, :end] = cap[:end]

    if permute:
        audio = audio.permute(0, 2, 1)
    return audio, torch.tensor(audio_lengths)

def batch_text(texts):
    """Merge captions, (from tuple of 1D tensor to 2D tensor). Pad with
    pad token."""
    char_lengths = [len(cap) for cap in texts]
    chars = torch.Tensor(len(texts), max(char_lengths)).long()
    chars.fill_(get_token_id(special_tokens.pad))
    for i, cap in enumerate(texts):
        end = char_lengths[i]
        chars[i, :end] = cap[:end]
    return chars, torch.tensor(char_lengths)

def batch_image(images):
    return torch.stack(images, 0)

def collate_fn(data, max_frames=2048):
    images, texts, audios = zip(* [(datum['image'],
                                    datum['text'],
                                    datum['audio']) for datum in data])
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = batch_image(images)
    audio, audio_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    return dict(image=images, audio=audio, text=chars, audio_len=audio_lengths,
                text_len=char_lengths)

def collate_fn_speech(data, max_frames=2048):
    texts, audios = zip(* [(datum['text'],
                            datum['audio']) for datum in data])
    audio, audio_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    return dict(audio=audio, text=chars, audio_len=audio_lengths,
                text_len=char_lengths)


def collate_fn_speech(data, max_frames=327680):
    texts, audios = zip(* [(datum['text'],
                            datum['audio']) for datum in data])
    audio, audio_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    return dict(audio=audio, text=chars, audio_len=audio_lengths,
                text_len=char_lengths)


def flickr8k_loader(root, meta_fname, language, feature_fname,
                    split='train', batch_size=32, shuffle=False,
                    max_frames=2048,
                    downsampling_factor=None):
    return torch.utils.data.DataLoader(
        dataset=Flickr8KData(root=root,
                             feature_fname=feature_fname,
                             meta_fname=meta_fname,
                             split=split,
                             language=language,
                             downsampling_factor=downsampling_factor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, max_frames=max_frames))


def librispeech_loader(root, meta_fname, feature_fname,
                       split='train', batch_size=32, shuffle=False,
                       max_frames=2048,
                       downsampling_factor=None):
    return torch.utils.data.DataLoader(
        dataset=LibriSpeechData(root=root,
                                feature_fname=feature_fname,
                                meta_fname=meta_fname,
                                split=split,
                                downsampling_factor=downsampling_factor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn_speech(x, max_frames=max_frames))


def spokencoco_loader(root, meta_fname, feature_fname,
                      split='train', batch_size=32, shuffle=False,
                      max_frames=2048,
                      downsampling_factor=None,
                      debug=False):
    return torch.utils.data.DataLoader(
        dataset=SpokenCOCOData(root=root,
                               feature_fname=feature_fname,
                               meta_fname=meta_fname,
                               split=split,
                               downsampling_factor=downsampling_factor,
                               debug=debug),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, max_frames=max_frames))

def raw_spokencoco_loader(dataset_path, metadata_path, audio_dir,
                          normalize=False,
                          sample_rate=16000,
                          split='train',
                          batch_size=32, shuffle=False,
                          max_frames=327680,
                          num_workers=8,
                          debug=None):
    return torch.utils.data.DataLoader(
        dataset=SpokenImageCaptionsDataset(data_path=dataset_path,
                                             metadata_path=metadata_path,
                                             audio_dir=audio_dir,
                                             normalize=normalize,
                                             sample_rate=sample_rate,
                                             split=split,
                                             debug=debug),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, max_frames=max_frames))


class SpokenImageCaptionsDataset(torch.utils.data.Dataset):
    """
    Dataset class that loads raw images and raw waveforms.
    Only work for SpokenCOCO so far.
    I think it'd be better to uniformize format between datasets rather than
    having a class for each dataset.
    """
    def __init__(self, data_path, metadata_path, audio_dir='', normalize=False, sample_rate=16000,
                 split='train', debug=None):
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        self.audio_dir = audio_dir
        self.split = split
        self.debug = debug
        self.load_metadata()
        self.normalize = normalize
        self.sample_rate = sample_rate

    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [sd[2] for sd in dataset.split_data]
        init_vocabulary(transcriptions)

    def load_metadata(self):
        if not self.metadata_path.is_file():
            raise FileNotFoundError("%s does not exist." % self.metadata_path)

        self.image_captions = {}
        self.split_data = []
        with open(self.metadata_path) as fmeta:
            metadata = json.load(fmeta)['data']
            if self.debug:
                metadata = metadata[:self.debug]

            # Load spoken captions / image pair
            for sample in metadata:
                img_id = sample['image']
                caption_list = []
                for caption in sample['captions']:
                    caption_list.append((caption['uttid'], caption['wav']))
                    self.split_data.append((img_id, caption['wav'], caption['text']))
                self.image_captions[img_id] = caption_list

    def postprocess_audio(self, audio, curr_sample_rate):
        # Handle multi-channels
        if audio.dim() == 2:
            audio = audio.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"Sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert audio.dim() == 1, audio.dim()

        if self.normalize:
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape)
        return audio

    def load_audio(self, audio_path):
        audio, curr_sample_rate = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio).float()
        audio = self.postprocess_audio(audio, curr_sample_rate)
        return audio

    def load_image(self, image_path):
        im = PIL.Image.open(image_path)


        # some functions such as taking the ten crop (four corners, center and
        # horizontal flip) normalise and resize.
        tencrop = transforms.TenCrop(224)
        tens = transforms.ToTensor()
        normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize = transforms.Resize(256, PIL.Image.ANTIALIAS)

        # there are some grayscale images in mscoco and places that the vgg and
        # resnet networks wont take
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im = tencrop(resize(im))
        im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
        return im

    def __getitem__(self, index):
        sd = self.split_data[index]
        image_id = sd[0]
        audio_id = sd[1]
        caption = sd[2]

        audio = self.load_audio(self.data_path / self.audio_dir / audio_id)
        image = self.load_image(self.data_path / image_id)
        text = caption2tensor(caption)

        return dict(image_id=image_id,
                    audio_id=audio_id,
                    image=image,
                    audio=audio,
                    text=text,
                    gloss=caption)

    def __len__(self):
        return len(self.split_data)

    def evaluation(self):
        """Returns image features, audio features, caption features, and a
        boolean array specifying whether a caption goes with an image."""
        audio = []
        text = []
        image = []
        matches = []
        image2idx = {}
        for sd in self.split_data:
            # Get ids
            image_id = sd[0]
            audio_id = sd[1]
            text_sample = sd[2]

            audio_sample = self.load_audio(self.data_path / self.audio_dir / audio_id)
            image_sample = self.load_image(self.data_path / image_id)

            # Add image
            if image_id in image2idx:
                image_idx = image2idx[image_id]
            else:
                image_idx = len(image)
                image2idx[image_id] = image_idx
                image.append(image_sample)

            # Add audio and text
            audio.append(audio_sample)
            text.append(text_sample)
            matches.append((len(audio) - 1, image_idx))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i, j in matches:
            correct[i, j] = True

        return dict(image=image, audio=audio, text=text, correct=correct)


class MyDataParallel(torch.nn.DataParallel):
    """
    Wrapper of DataParallel to allow having access to class attributes.
    See : https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
