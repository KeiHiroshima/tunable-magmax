import hashlib
import json
import os
import os.path
from logging import getLogger
from shutil import rmtree

import torch
from torchvision.datasets import ImageFolder


logger = getLogger(__name__)

SUBSET_CONFIG_FILE = "config.json"


class _ImageNetR:
    """
    Internal implementation of ImageNet-R with domain/class subsetting.

    Reads an existing train/test split (<root>/imagenet-r/{train,test}) through
    a subset of symlinks (<root>/imagenet-r_subsets/<md5>/{train,test}) that
    build_subset creates beforehand.
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

    @classmethod
    def default_subset_config(cls):
        return {"domains": cls.DOMAINS, "classes": cls.CLASSES}

    @staticmethod
    def split_dir(root, train):
        """Where the train/test split lives: <root>/imagenet-r/{train,test}."""
        return os.path.join(os.path.abspath(os.path.expanduser(root)), "imagenet-r",
                            "train" if train else "test")

    @classmethod
    def require_split(cls, root):
        """Fail unless the train/test split is already in place.

        The split is part of the experiment: the paper's checkpoints were
        trained on one particular (unseeded) 80:20 split, so it has to be
        reused as it is. This code used to make a new split, moving files,
        whenever it found none; a wrong data path then silently produced a
        different split. Now it refuses instead.
        """
        missing = [d for d in (cls.split_dir(root, True), cls.split_dir(root, False))
                   if not os.path.isdir(d)]
        if missing:
            raise RuntimeError(
                f"ImageNet-R train/test split not found: {', '.join(missing)}. "
                "Place the existing split there (<data>/imagenet-r/{train,test}/<class>/); "
                "it is not created automatically."
            )

    @classmethod
    def subset_path(cls, root, subset_config, train):
        return os.path.join(
            os.path.abspath(os.path.expanduser(root)),
            "imagenet-r_subsets",
            cls.get_md5(subset_config),
            "train" if train else "test",
        )

    @staticmethod
    def is_subset_complete(path):
        """config.json is written last, so a subset without it is unfinished."""
        return os.path.isfile(os.path.join(path, SUBSET_CONFIG_FILE))

    @classmethod
    def build_subset(cls, root, subset_config=None, train=True):
        """Create the subset of symlinks for one split, if it is not there yet.

        Run once before any experiment (src/datasets/prepare_imagenetr.py),
        never from the training or merging jobs: those run in parallel and
        would race on the same directory. The subset is built in a temporary
        directory and renamed into place only when complete, so an
        interrupted build leaves nothing that looks finished. An unfinished
        subset left by older code (no config.json) is built again.

        Returns the subset's path.
        """
        cls.require_split(root)
        subset_config = subset_config or cls.default_subset_config()
        path = cls.subset_path(root, subset_config, train)
        if cls.is_subset_complete(path):
            logger.info(f"Using subset that already exists: {path}")
            return path
        if os.path.exists(path):
            logger.warning(f"Rebuilding unfinished subset {path}")
            rmtree(path)  # symlinks only; rmtree does not follow them

        source = cls.split_dir(root, train)
        tmp = f"{path}.tmp-{os.getpid()}"
        if os.path.exists(tmp):
            rmtree(tmp)
        for c in subset_config["classes"]:
            source_cls_path = os.path.join(source, c)
            new_cls_path = os.path.join(tmp, c)
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
                    f"Samples from domains {subset_config['domains']} do not exist "
                    f"for class {c}, leaving an empty folder!"
                )
        with open(os.path.join(tmp, SUBSET_CONFIG_FILE), "w") as config_file:
            json.dump(subset_config, config_file)
        os.rename(tmp, path)
        logger.info(f"Created subset {path}")
        return path

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        subset_config=None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.require_split(self.root)

        subset_config = subset_config or self.default_subset_config()
        assert "domains" in subset_config and "classes" in subset_config
        logger.debug(f"Using subset config: {subset_config}")

        path = self.subset_path(self.root, subset_config, train)
        if not self.is_subset_complete(path):
            raise RuntimeError(
                f"ImageNet-R subset {path} is missing or unfinished. Create it first with "
                "`uv run python -m src.datasets.prepare_imagenetr` "
                "(not from the experiment jobs, which run in parallel)."
            )
        self.fpath = path
        self.data = ImageFolder(self.fpath, transform=transform, allow_empty=True)


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
