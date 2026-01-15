# Installation

Requires uproot, h5py, and numpy.

# Usage

```bash
python root_to_hdf5.py infiles outfolder
```

Will convert all root files in `infiles` to HDF5 files. More options can be seen with the `-h` flag.

HDF5 files are named the same as the input root files, but with ".root" replaced with ".hdf5".
