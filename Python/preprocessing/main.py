"""Entry point for the OmniPhotos desktop launcher and headless pipeline."""

import argparse
import multiprocessing as mp


def parse_bool(value):
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config_file", help="preprocessing YAML file")
    parser.add_argument("--input", dest="input_path", help="override the input with a stitched video or image folder")
    parser.add_argument("-hd", "--headless", nargs="?", const=True, default=False, type=parse_bool,
                        help="process in the terminal and exit with the job status")
    args = parser.parse_args()
    if args.headless:
        if not args.config_file:
            parser.error("--headless requires --config_file")
        from preproc_app import PreprocAPP

        PreprocAPP(vars(args)).run()
    else:
        from desktop import launch

        launch(args.config_file, args.input_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
