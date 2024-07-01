from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from .utils import get_positional_encoding, get_positional_fourier_encoding
from .transforms import Compose, TWrapper
from time import time as t
import os

__all__ = ['list_datasets', 'Dataset']

def list_datasets(root="./data"):
	return [n.stem for n in Path(root).iterdir() if n.is_dir()]
	
class Dataset(torch.utils.data.Dataset):
    """Dataset class for image compression datasets."""

    def __init__(self, name, path=None, transform=None, debug=False, return_filename=False, train=False, **kwargs):
        """
        Args:
            name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
            path (str): if given the dataset is loaded from path instead of by name.
            transforms (Transform): transforms to apply to image
            debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
        """
        super().__init__()
        
        self.train = train

        if path is None:
            assert len(name.split(
                '#')) == 2, f'invalid name ({name}). correct template: ds_name#ds_type. No "#" in ds_name or ds_type allowed'

            ds_name = name.split('#')[0]
            ds_type = name.split('#')[1]
            path = data_zoo_stereo[ds_name][ds_type]
        else:
            # If dataset has been specified and no path, use path manager
            # If path has been specified with or without dataset name, use the path
            path = Path(path)
            if not path.exists:
                raise OSError("The path {} doesn't exist".format(path))
            if not path.is_dir:
                raise OSError("The path {} is not a directory".format(path))
        self.path = path

        self.name = name
        self.return_filename = return_filename
        self.ds_name = name.split('#')[0]
        self.ds_type = name.split('#')[1]
        self.transform = transform or [(lambda x: x)]
        self.transforms = transforms.Compose([*self.transform])
        self.file_dict = self.get_file_dict()
        if debug:
            for k in self.file_dict:
                self.file_dict[k] = self.file_dict[k][:10]

        print(f'Loaded dataset {name} from {path}. Found {len(self.file_dict["left_image"])} files.')

    def __len__(self):
        return len(self.file_dict[list(self.file_dict.keys())[0]])

    # Not very secure, assumes only torch.seed is relevant
    def apply_transforms(self, data_dict: dict):
        seed = torch.seed()

        for k in ('left', 'right'):
            #torch.manual_seed(seed)

            #data_dict[k] = self.transforms(data_dict[k])
            data_dict[k] = transforms.ToTensor()(data_dict[k])

        return data_dict

    def __getitem__(self, idx):
        _, tl, thl, phl, __, ___, ____ = str(self.file_dict['left_image'][idx]).split("_")
        _, tr, thr, phr, __, ___, ____ = str(self.file_dict['right_image'][idx]).split("_")
        ttl, thl, phl, ttr, thr, phr = float(tl)/30., .5*(float(thl)/90.+1.), .5*(float(phl)/180.+1.), float(tr)/30., .5*(float(thr)/90.+1.), .5*(float(phr)/180.+1)
        
        data_dict = {
            'left': Image.open(self.file_dict['left_image'][idx]),
            'right': Image.open(self.file_dict['right_image'][idx]),
            'left_param': tl,
            'right_param': tr,
            'pl': torch.tensor([[ttl, thl, phl]]),
            'pr': torch.tensor([[ttr, thr, phr]])
        }
        #print(data_dict['left_param'])
        data_dict = self.apply_transforms(data_dict)

        if self.return_filename:
            return data_dict, str(self.file_dict['left_image'][idx])
        else:
            return data_dict, idx

    def get_file_dict(self) -> list:
        """Get dictionary of all files in folder data_dir."""

        if self.ds_name == 'cityscapes':
            image_list = [file for file in self.path.glob(
                '**/*') if file.is_file() and file.suffix.lower() == '.png']

            # set removes duplicates due to *_disparity.png, *_rightImg8bit.png, *_leftImg8bit.png
            names = list(
                {'_'.join(str(f).split('_')[:-1]) for f in image_list})
            names.sort()

            files = {
                'left_image': [name + '_leftImg8bit.png' for name in names],
                'right_image': [name + '_rightImg8bit.png' for name in names],
                'disparity_image': [name + '_disparity.png' for name in names]
            }
        elif self.ds_name == 'instereo2k':
            folders = [f for f in self.path.iterdir() if f.is_dir()]
            left_images = [f / 'left.png' for f in folders]
            right_images = [f / 'right.png' for f in folders]

            files = {
                'left_image': left_images,
                'right_image': right_images
            }

        #elif self.ds_name == 'vorts-v812-t30':
            #path1 = Path('datasets/vorts-vol-v812-t15/')
            #path2 = Path('datasets/vorts-vol-v812-t16-30/')
            #image_list = [file for file in path1.glob(
             #   '**/*') if file.is_file() and file.suffix.lower() == '.png'] + [file for file in path2.glob(
            #    '**/*') if file.is_file() and file.suffix.lower() == '.png']

            #image_list.sort(key=lambda y: (float(str(y).split("_")[1]), float(str(y).split("_")[2]), float(str(y).split("_")[3])))
            #left_images = image_list[::2]
            #right_images = image_list[1::2]
            #files = {
             #   'left_image': left_images,
             #   'right_image': right_images
            #}
        else:
            # Write your own section here
            # The folder path can be accessed via self.path
            # The result should be a dictionary 'files' with the keys 'left_image' and 'right_image' and corresponding lists with filenames:
            # files = {
            # 	'left_image': [(self.path / 'left_1.png'), (self.path / 'left_2.png')],
            # 	'right_image': [(self.path / 'right_1.png'), (self.path / 'right_2.png')]
            # }
            #print(self.path)
            image_list = [file for file in self.path.glob(
                '**/*') if file.is_file() and file.suffix.lower() == '.png']
            
           # print(self.path)
           # print(image_list)
            
            image_list.sort(key=lambda y: (float(str(y).split("_")[1]), float(str(y).split("_")[2]), float(str(y).split("_")[3])))
            left_images = image_list[::4] if self.train else image_list[::2]
            right_images = image_list[1::4] if self.train else image_list[1::2]
            files = {
                'left_image': left_images,
                'right_image': right_images
            }
            
            # else:
            # raise NotImplementedError
            f = open('pair.txt', 'w')
            for i in range(len(left_images)):
                f.write(str(left_images[i]).split("/")[-1]+'    '+str(right_images[i]).split("/")[-1]+'\n')
            f.close()
        return files