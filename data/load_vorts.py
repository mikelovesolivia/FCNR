from pathlib import Path
import argparse

if __name__ == "__main__":
    path = Path('./data/vorts-vol-v92-t30')
    folder = Path('./vorts-vol-v92-t30-folder/')
    folder.mkdir(exist_ok=True, parents=False)

    image_list = [file for file in path.glob('**/*') if file.is_file() and file.suffix.lower() == '.png']

    image_list.sort(
        key=lambda y: (float(str(y).split("_")[1]), float(str(y).split("_")[2]), float(str(y).split("_")[3])))

    files = []
    left_images = image_list[::2]
    # mid_images = image_list[1::3]
    right_images = image_list[1::2]
    files = {f'{l}, {r}' for l, r in zip(left_images, right_images)}
    for t in ['train', 'test']:
        print(f'writing {t} files to {folder}')
        with open(folder / f'{t}.txt', 'w') as text_file:
            text_file.write('\n'.join(files))


