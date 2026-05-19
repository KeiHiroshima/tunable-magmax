import hashlib
import json
import logging
import os
import os.path
import random
from logging import getLogger
from shutil import move, rmtree

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from src.datasets.common import get_subset_indices_with_classes, get_task_classes

logger = getLogger(__name__)


class _ImageNetR:
    """
    Internal implementation of ImageNet-R dataset with domain/class subsetting and automatic train/test splitting.
    """

    DOMAINS = [
        "deviantart",
        "cartoon",
        "misc",
        "embroidery",
        "videogame",
        "painting",
        "art",
        "graphic",
        "origami",
        "tattoo",
        "toy",
        "sticker",
        "graffiti",
        "sculpture",
        "sketch",
    ]
    CLASSES = [
        "African_chameleon",
        "Granny_Smith",
        "accordion",
        "acorn",
        "afghan_hound",
        "ambulance",
        "american_egret",
        "ant",
        "assault_rifle",
        "axolotl",
        "baboon",
        "backpack",
        "badger",
        "bagel",
        "bald_eagle",
        "banana",
        "barn",
        "baseball_player",
        "basketball",
        "basset_hound",
        "bathtub",
        "beagle",
        "beaver",
        "bee",
        "beer_glass",
        "bell_pepper",
        "binoculars",
        "birdhouse",
        "bison",
        "black_swan",
        "bloodhound",
        "border_collie",
        "boston_terrier",
        "bow_tie",
        "boxer",
        "broccoli",
        "broom",
        "bucket",
        "burrito",
        "cabbage",
        "candle",
        "cannon",
        "canoe",
        "carousel",
        "castle",
        "cauldron",
        "centipede",
        "cheeseburger",
        "cheetah",
        "chihuahua",
        "chimpanzee",
        "chow_chow",
        "clown_fish",
        "cobra",
        "cocker_spaniels",
        "cockroach",
        "collie",
        "cowboy_hat",
        "cucumber",
        "dalmatian",
        "dragonfly",
        "duck",
        "eel",
        "electric_guitar",
        "espresso",
        "fire_engine",
        "flamingo",
        "flute",
        "fly",
        "fox_squirrel",
        "french_bulldog",
        "gasmask",
        "gazelle",
        "german_shepherd_dog",
        "gibbon",
        "golden_retriever",
        "goldfinch",
        "goldfish",
        "goose",
        "gorilla",
        "grand_piano",
        "grasshopper",
        "great_white_shark",
        "grey_whale",
        "guillotine",
        "guinea_pig",
        "hammer",
        "hammerhead",
        "harmonica",
        "harp",
        "hatchet",
        "hen",
        "hermit_crab",
        "hippopotamus",
        "hotdog",
        "hummingbird",
        "husky",
        "hyena",
        "ice_cream",
        "iguana",
        "italian_greyhound",
        "jeep",
        "jellyfish",
        "joystick",
        "junco",
        "killer_whale",
        "king_penguin",
        "koala",
        "lab_coat",
        "labrador_retriever",
        "ladybug",
        "lawn_mower",
        "lemon",
        "leopard",
        "lighthouse",
        "lion",
        "lipstick",
        "llama",
        "lobster",
        "lorikeet",
        "mailbox",
        "mantis",
        "meerkat",
        "military_aircraft",
        "missile",
        "mitten",
        "mobile_phone",
        "monarch_butterfly",
        "mushroom",
        "newt",
        "orangutan",
        "ostrich",
        "panda",
        "parachute",
        "peacock",
        "pelican",
        "pembroke_welsh_corgi",
        "pickup_truck",
        "pig",
        "pineapple",
        "pirate_ship",
        "pizza",
        "polar_bear",
        "pomegranate",
        "pomeranian",
        "porcupine",
        "pretzel",
        "puffer_fish",
        "pug",
        "red_fox",
        "revolver",
        "rottweiler",
        "rugby_ball",
        "saint_bernard",
        "sandal",
        "saxophone",
        "scarf",
        "school_bus",
        "schooner",
        "scorpion",
        "scottish_terrier",
        "scuba_diver",
        "sea_lion",
        "shield",
        "shih_tzu",
        "skunk",
        "snail",
        "snow_leopard",
        "soccer_ball",
        "space_shuttle",
        "spider_web",
        "standard_poodle",
        "starfish",
        "steam_locomotive",
        "stingray",
        "strawberry",
        "submarine",
        "tabby_cat",
        "tank",
        "tarantula",
        "tennis_ball",
        "tiger",
        "timber_wolf",
        "toucan",
        "toy_poodle",
        "tractor",
        "tree_frog",
        "trombone",
        "vase",
        "violin",
        "volcano",
        "vulture",
        "weimaraner",
        "west_highland_white_terrier",
        "wheelbarrow",
        "whippet",
        "wine_bottle",
        "wood_rabbit",
        "yorkshire_terrier",
        "zebra",
    ]

    @classmethod
    def get_md5(cls, subset_config):
        return hashlib.md5(json.dumps(subset_config).encode("utf-8")).hexdigest()

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        subset_config=None,
        download=False,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.fpath = os.path.join(
            os.path.abspath(root), "imagenet-r", "train" if train else "test"
        )

        self.download_and_split_if_needed(download)

        if not subset_config:
            subset_config = {"domains": self.DOMAINS, "classes": self.CLASSES}
        assert "domains" in subset_config and "classes" in subset_config
        logger.debug(f"Using subset config: {subset_config}")

        id = self.get_md5(subset_config)
        new_fpath = os.path.join(
            os.path.abspath(root),
            "imagenet-r_subsets",
            id,
            "train" if train else "test",
        )
        if not os.path.exists(new_fpath):
            print("Creating a subset from scratch")
            for cls in subset_config["classes"]:
                source_cls_path = os.path.join(self.fpath, cls)
                new_cls_path = os.path.join(new_fpath, cls)
                os.makedirs(new_cls_path, exist_ok=True)

                for domain in subset_config["domains"]:
                    for img_name in os.listdir(source_cls_path):
                        if img_name.startswith(domain):
                            os.symlink(
                                os.path.join(source_cls_path, img_name),
                                os.path.join(new_cls_path, img_name),
                            )

                if not os.listdir(new_cls_path):
                    logger.warning(
                        f"Samples from domains {subset_config['domains']} do not exist for class {cls}, leaving an empty folder!"
                    )
                    # print(f"Samples from domains {subset_config['domains']} do not exist for class {cls}, skipping!")
                    # os.rmdir(new_cls_path)

            with open(os.path.join(new_fpath, "config.json"), "w") as config_file:
                json.dump(subset_config, config_file)
        else:
            logger.info("Using subset that already exists")

        self.fpath = new_fpath
        self.data = ImageFolder(self.fpath, transform=transform, allow_empty=True)

    def download_and_split_if_needed(self, download):
        """Download dataset if missing and split into train/test (80:20) if not already split."""
        url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        filename = "imagenet-r.tar"

        fpath = os.path.join(self.root, "imagenet-r")

        if not os.path.exists(fpath):
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )
            else:
                print("Downloading from " + url)
                download_url(url, self.root, filename=filename)

                import tarfile

                tar_ref = tarfile.open(os.path.join(self.root, filename), "r")
                tar_ref.extractall(self.root)
                tar_ref.close()

                # rename dirs, e.g. "n01443537" -> "goldfish"
                with open(fpath + "/README.txt", "r") as f:
                    for line in f:
                        if not line.startswith("n"):
                            continue
                        curr_dir = os.path.join(fpath, line.split(" ")[0])
                        new_dir = os.path.join(fpath, line.split(" ")[1].split("\n")[0])
                        move(curr_dir, new_dir)

        if not os.path.exists(os.path.join(fpath, "train")) and not os.path.exists(
            os.path.join(fpath, "test")
        ):
            dataset = datasets.ImageFolder(fpath, transform=None)

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices

            self.train_file_list = [dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [dataset.imgs[i][0] for i in val_idx]

            self._split(dataset, fpath)

    def _split(self, dataset, fpath):
        train_folder = os.path.join(fpath, "train")
        test_folder = os.path.join(fpath, "test")

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))

        for path in self.train_file_list:
            if "\\" in path:
                path = path.replace("\\", "/")
            src = path
            dst = os.path.join(train_folder, "/".join(path.split("/")[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if "\\" in path:
                path = path.replace("\\", "/")
            src = path
            dst = os.path.join(test_folder, "/".join(path.split("/")[-2:]))
            move(src, dst)

        for c in dataset.classes:
            path = os.path.join(fpath, c)
            rmtree(path)


class ImageNetR:
    BASE_CLASS = _ImageNetR
    default_class_order = [
        186,
        157,
        67,
        59,
        117,
        1,
        122,
        14,
        131,
        40,
        66,
        199,
        92,
        170,
        197,
        118,
        173,
        37,
        146,
        191,
        4,
        198,
        64,
        56,
        51,
        165,
        0,
        107,
        175,
        148,
        187,
        161,
        137,
        180,
        133,
        181,
        177,
        12,
        20,
        74,
        143,
        142,
        110,
        116,
        194,
        138,
        183,
        89,
        45,
        121,
        190,
        94,
        36,
        83,
        130,
        120,
        21,
        38,
        125,
        145,
        182,
        103,
        105,
        61,
        32,
        141,
        13,
        112,
        111,
        104,
        26,
        69,
        39,
        156,
        147,
        108,
        72,
        10,
        96,
        188,
        16,
        63,
        152,
        86,
        160,
        119,
        184,
        7,
        172,
        132,
        44,
        42,
        82,
        15,
        114,
        127,
        85,
        168,
        55,
        65,
        3,
        95,
        171,
        176,
        33,
        43,
        17,
        97,
        163,
        62,
        75,
        123,
        128,
        124,
        99,
        29,
        28,
        166,
        41,
        35,
        76,
        77,
        5,
        84,
        102,
        60,
        78,
        70,
        164,
        113,
        30,
        91,
        81,
        18,
        155,
        179,
        73,
        46,
        80,
        31,
        150,
        79,
        174,
        153,
        50,
        144,
        9,
        167,
        134,
        135,
        101,
        49,
        106,
        154,
        23,
        100,
        48,
        25,
        71,
        139,
        162,
        158,
        159,
        87,
        90,
        11,
        24,
        57,
        19,
        52,
        129,
        140,
        178,
        53,
        169,
        195,
        126,
        47,
        151,
        136,
        22,
        58,
        34,
        8,
        185,
        54,
        109,
        193,
        68,
        192,
        149,
        115,
        98,
        88,
        93,
        6,
        196,
        189,
        2,
        27,
    ]
    class_order_dict = {
        "A": default_class_order,
    }
    default_domain_order = [9, 14, 7, 5, 6, 8, 13, 12, 1, 2, 10, 3, 4, 11, 0]

    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
        subset_config=None,
        args=None,
        **kwargs,
    ):
        self.train_dataset = _ImageNetR(
            location,
            train=True,
            transform=preprocess,
            target_transform=None,
            download=False,
            subset_config=subset_config,
        ).data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _ImageNetR(
            location,
            train=False,
            transform=preprocess,
            target_transform=None,
            download=False,
            subset_config=subset_config,
        ).data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.classnames = [
            c.replace("_", " ") for c in list(self.train_dataset.class_to_idx.keys())
        ]

        if args and args.taskseq_pattern not in self.class_order_dict:
            raise ValueError(f"Unknown taskseq_pattern: {args.taskseq_pattern}")
        self.default_class_order = self.class_order_dict[
            args.taskseq_pattern if args else "A"
        ]

    def _construct_train_subset_each_task(
        self,
        n_splits: int,
        num_train_data_each_task: int,
        seed: int = 42,
    ):
        """
        Constructs a list of subsets of the ImageNet-R train dataset, one for each task.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_subsets = []
        for split_idx in range(n_splits):
            task_classes = get_task_classes(
                self.default_class_order, n_splits, split_idx
            )
            # a bit confusing, but self.train_dataset is ImageFolder, which doesn't have targets
            # but its parent class, DatasetFolder, does. And it's populated.
            task_indices = get_subset_indices_with_classes(
                self.train_dataset, task_classes
            )

            num_train_data_each_task = min(len(task_indices), num_train_data_each_task)
            sampled_task_indices = np.random.choice(
                task_indices, num_train_data_each_task, replace=False
            )
            train_subsets.append(Subset(self.train_dataset, sampled_task_indices))

        return train_subsets

    def _construct_target_imagenetr_dataset(
        self,
        num_data: int,
        n_splits: int,
        num_data_from_tasks: list = None,
        ratio_data_from_task: list = None,
        seed: int = 42,
    ):
        """
        Constructs a subset of the ImageNet-R test dataset by sampling from each task.
        """
        np.random.seed(seed)

        # Determine the number of samples to draw from each task
        if num_data_from_tasks is not None:
            if len(num_data_from_tasks) != n_splits:
                raise ValueError("Length of num_data_from_tasks must match n_splits.")
            if sum(num_data_from_tasks) != num_data:
                logging.warning(
                    f"Sum of num_data_from_tasks ({sum(num_data_from_tasks)}) does not equal num_data ({num_data})."
                )
            samples_per_task = num_data_from_tasks

        elif ratio_data_from_task is not None:
            # assign num task ramdomly
            num_data_each_task = [0] * n_splits
            task_idx_selected = random.sample(
                range(n_splits), len(ratio_data_from_task)
            )
            num_data_to_be_selected = (
                np.floor(np.array(ratio_data_from_task) * num_data).astype(int).tolist()
            )
            # adjust last element to match total num_data

            remainder = num_data - sum(num_data_to_be_selected)
            for _ in range(remainder):
                num_data_to_be_selected[
                    np.random.randint(0, len(num_data_to_be_selected))
                ] += 1

            num_data_to_be_selected_shuffled = np.random.permutation(
                num_data_to_be_selected
            )

            for i, task_idx in enumerate(task_idx_selected):
                num_data_each_task[task_idx] = num_data_to_be_selected_shuffled[i]

            logger.info(
                f"task_idx_selected: {task_idx_selected}, num_data_each_task: {num_data_each_task}"
            )
            logger.debug(f"sum: {sum(num_data_each_task)}, num_data: {num_data}")

        else:
            # uniform distribution
            num_data_each_task = [num_data // n_splits] * n_splits
            remainder = num_data % n_splits
            for i in range(remainder):
                num_data_each_task[np.random.randint(0, n_splits)] += 1

        meta_data_indices = []
        test_data_list = []
        for split_idx, num_samples in zip(range(n_splits), num_data_each_task):
            if num_samples == 0:
                test_data_list.append(None)
                continue
            else:
                task_classes = get_task_classes(
                    self.default_class_order, n_splits, split_idx
                )
                task_indices = get_subset_indices_with_classes(
                    self.test_dataset, task_classes
                )

                num_samples = min(len(task_indices), num_samples)
                """if len(task_indices) < num_samples:
                    raise ValueError(
                        f"Task {split_idx} (classes {task_classes}) has only {len(task_indices)} test samples, "
                        f"but {num_samples} were requested."
                    )"""

                sampled_task_indices = np.random.choice(
                    task_indices, num_samples, replace=False
                )

                num_meta = int(0.1 * len(sampled_task_indices))

                meta_data_indices.extend(sampled_task_indices[:num_meta])
                test_data_list.append(
                    Subset(self.test_dataset, sampled_task_indices[num_meta:])
                )

                logging.debug(
                    f"Task {split_idx} selected {num_samples} samples from classes {len(task_classes)}\\nmeta data size: {num_meta}, test data size: {num_samples - num_meta}."
                )

        np.random.shuffle(meta_data_indices)
        meta_data = Subset(self.test_dataset, meta_data_indices)

        # update num_data_each_task
        num_data_each_task = [len(td) if td is not None else 0 for td in test_data_list]

        return (
            task_idx_selected,
            num_data_each_task,
            meta_data,
            test_data_list,
        )
