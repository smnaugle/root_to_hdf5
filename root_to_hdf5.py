import argparse
import logging
from pathlib import Path
from typing import TypedDict, cast

import h5py
import numpy as np
import uproot as up
from uproot.behaviors.TBranch import HasBranches

logger = logging.getLogger(__name__)


class ProcessOptions(TypedDict):
    compression: str | None
    compression_level: str | None


def check_outdir(outdir: str) -> None:
    outpath = Path(outdir)
    if not outpath.is_dir():
        raise NotADirectoryError(f"Supplied path `{outdir}` is not a directory.")


def check_infiles(infiles: str) -> None:
    if len(infiles) == 0:
        logger.warning(f"No root files found under {infiles}")
    for p in infiles:
        path = Path(p)
        if not (path.is_file() and path.suffix == ".root"):
            raise RuntimeError(f"{path} is not a root file.")


def setup_outfile(outfilename: str, intree: HasBranches, branches: list[str], options: ProcessOptions) -> h5py.File:
    if Path(outfilename).exists():
        raise FileExistsError(f"File {outfilename} already exists. Please delete.")
    outfile = h5py.File(outfilename, "w")
    for array in up.iterate(intree, branches, library="numpy"):
        for a in array:
            outfile.create_dataset(
                "/" + a,
                shape=(0,),
                maxshape=(None,),
                dtype=array[a].dtype,
                compression=options["compression"],
                compression_opts=options["compression_level"],
            )
        break
    return outfile


def process_chunk(array: dict[str, np.ndarray], outfile: h5py.File, options: ProcessOptions):
    # Currently unused, but might be useful later
    _ = options

    def write(dset: h5py.Dataset, data: np.ndarray):
        data_len = len(data)
        dset.resize(dset.shape[0] + data_len, axis=0)
        dset[dset.shape[0] - data_len :] = data

    for branch in array:
        dset = cast(h5py.Dataset, outfile[branch])
        write(dset, array[branch])


def process_file(infile: str, outdir: str, branches: list[str], tree: str, options: ProcessOptions):
    file = up.open(infile + ":" + tree)
    inpath = Path(infile).resolve().stem
    outfilename = str(Path(outdir).resolve()) + "/" + inpath + ".hdf5"
    if not issubclass(type(file), HasBranches):
        raise TypeError("Expected up.open to return TTree, did you provide the right tree name?")
    file = cast(HasBranches, file)
    if len(branches) == 0:
        branches = []
        for branch in file.branches:
            name = branch.member("fName")
            branches.append(name)
    outfile = setup_outfile(outfilename, file, branches, options)
    for array in up.iterate(file, branches, library="numpy"):
        array = cast(dict[str, np.ndarray], array)
        process_chunk(array, outfile, options)


def main():
    desc = """
Will convert all root files in `infiles` to HDF5 files.

Example: `python root_to_hdf5.py rootfiles/*.root outdir/`

HDF5 files are named the same as the input root files, but with ".root" replaced with ".hdf5".

Requires uproot, h5py, and numpy.
    """
    args = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument(
        "infiles",
        nargs="+",
        help="Root file(s) to convert",
    )
    args.add_argument("outfolder", type=str, help="Folder in which to output files")
    args.add_argument(
        "-b",
        "--branches",
        action="append",
        default=[],
        help="Branches to include in outfiles. Leave unset to include all branches.",
    )
    args.add_argument(
        "-t",
        "--tree",
        default="",
        help="Tree in root file to find the branches.",
    )
    args.add_argument(
        "-l",
        "--log",
        type=str,
        default="WARNING",
        help="Logging level. [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
    )
    args.add_argument(
        "-x",
        "--compression",
        type=str,
        default=None,
        help="Compression. [gzip, lzf]",
    )
    args.add_argument(
        "-v",
        "--level",
        type=int,
        default=None,
        help="Compression level. Only applicable to gzip compression. Levels 0-9.",
    )
    parsed = args.parse_args()

    logging.basicConfig(level=parsed.log.upper())

    check_outdir(parsed.outfolder)
    check_infiles(parsed.infiles)

    for i, infile in enumerate(parsed.infiles):
        logger.info(f"On file {infile}. File {i + 1} out of {len(parsed.infiles)}")
        try:
            process_file(
                infile,
                parsed.outfolder,
                parsed.branches,
                parsed.tree,
                {
                    "compression": parsed.compression,
                    "compression_level": parsed.level,
                },
            )
        except OSError as e:
            logger.warning(f"Could not process {infile}.")
            logger.warning(f"Recieved: {type(e)}, {e}")
            logger.warning(f"Skipping {infile}.")


if __name__ == "__main__":
    main()
