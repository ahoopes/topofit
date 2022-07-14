import itertools
import numpy as np
import surfa as sf
import torch

from . import ico
from . import utils


# image shape used during training
target_image_shape = (96, 144, 192)


def load_subject_data(subj, hemi, ground_truth=False, low_res=False, vol="norm.mgz"): 
    """
    Load a FreeSurfer subject image and surface. Use the talairach alignment
    to place the initial template surface and crop the image.
    """

    # load bias corrected image and talairach affine
    image = sf.load_volume(f'{subj}/mri/{vol}')
    affine = sf.load_affine(f'{subj}/mri/transforms/talairach.xfm.lta').inv().convert(space='vox', target=image)

    # load the initial template surface and align to subject
    template = ico.get_initial_template(hemi)
    template.vertices = affine.transform(template.vertices)
    template.geom = image

    # generate an image cropped based on the aligned template surface
    cropping = compute_image_cropping(image.baseshape, template.vertices)
    cropped_image = image[cropping].reshape(target_image_shape)

    # normalize image values
    cropped_image = cropped_image.astype(np.float32) / cropped_image.percentile(99.99, nonzero=True)

    # convert and map template vertices to a lower resolution ico-surface
    input_vertices = template.convert(space='vox', geometry=cropped_image).vertices.astype(np.float32)
    input_vertices = input_vertices[ico.get_mapping(6, 1)]

    # build a data dictionary to return
    data = {
        'input_image': torch.from_numpy(cropped_image.data),
        'input_vertices': torch.from_numpy(input_vertices),
        'input_geometry': image.geom,
        'cropped_geometry': cropped_image.geom,
    }

    # ground-truths might be needed (for training)
    if ground_truth:
        true_vertices = sf.load_mesh(f'{subj}/surf/{hemi}.white.ico.surf')
        true_vertices = true_vertices.convert(space='vox', geometry=cropped_image).vertices.astype(np.float32)
        if low_res:
            true_vertices = true_vertices[ico.get_mapping(7, 6)]
        data['true_vertices'] = torch.from_numpy(true_vertices)
    
    return data


def compute_image_cropping(image_shape, vertices):
    """
    Compute the correct image cropping given the bounding box of aligned vertices
    """
    vmin = vertices.min(0)
    vmax = vertices.max(0)

    image_limit = np.asarray(image_shape) - 1
    pmin = np.clip(np.floor(vertices.min(0)), (0, 0, 0), image_limit)
    pmax = np.clip(np.ceil(vertices.max(0)),  (0, 0, 0), image_limit)

    pdiff = np.asarray(target_image_shape) - (pmax - pmin)
    if np.any(pdiff < 0):
        raise RuntimeError('alignment exceeds target image shape')

    # pad is necessary
    padding = pdiff / 2.0
    pmin = np.clip(pmin - np.floor(padding), (0, 0, 0), image_limit)
    pmax = np.clip(pmax + np.ceil(padding), (0, 0, 0), image_limit)
    source_shape = pmax - pmin
    cropping = tuple([slice(int(a), int(b)) for a, b in zip(pmin, pmax)])
    return cropping


class InfiniteSampler(torch.utils.data.IterableDataset):
    """
    Iterable torch dataset that infinitively samples training subjects.
    """
    def __init__(self, hemi, training_subjs, low_res):
        super().__init__()
        self.hemi = hemi
        self.training_subjs = training_subjs
        self.low_res = low_res

    def __iter__(self):
        yield from itertools.islice(self.infinite(), 0, None, 1)

    def infinite(self):
        while True:
            idx = np.random.randint(len(self.training_subjs))
            subj = self.training_subjs[idx]
            try:
                data = load_subject_data(subj, self.hemi, ground_truth=True, low_res=self.low_res)
                data = {k: v for k, v in data.items() if k in ('input_image', 'input_vertices', 'true_vertices')}
            except RuntimeError:
                continue
            yield from [data]


class Collator:

    def __init__(self, data):
        self.data = data[0]

    def pin_memory(self):
        for key, value in self.data.items():
            self.data[key] = value.pin_memory()
        return self


def get_data_loader(hemi, training_subjs, low_res=False, prefetch_factor=8):
    collate_fn = lambda batch : Collator(batch)
    sampler = InfiniteSampler(hemi, training_subjs, low_res)
    data_loader = torch.utils.data.DataLoader(sampler, batch_size=1, num_workers=1,
        prefetch_factor=prefetch_factor, collate_fn=collate_fn, pin_memory=True)
    return data_loader
