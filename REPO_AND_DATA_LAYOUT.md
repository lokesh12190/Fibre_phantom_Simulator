# Repository and data layout

This directory is the complete bundle: code plus the generated datasets.
Copy the whole directory to transfer everything.

## Why the data is not in git

`data/` is excluded from version control. A single projection stack is about
6 GB and one dataset is 26 GB, well beyond what git handles without LFS. The
datasets are transferred by copying this directory; only the code, configs and
docs are tracked in git.

## Why the git repository is not in this directory

This directory lives on a CIFS/SMB network share mounted with a fixed file
mode and root ownership, so no process can `chmod` here. Git requires `chmod`
for every config write, so `git init` fails on this filesystem with:

    error: chmod on .git/config.lock failed: Operation not permitted

The repository therefore lives on local disk instead. Nothing is lost by the
split: because `data/` is git-ignored anyway, the repository only ever
contained the code, configs and docs, and those are kept byte-identical to the
copies here.

If you are setting this up elsewhere on a normal filesystem, this split is
unnecessary — just `git init` in place.

## Publishing

    cd <repository directory on local disk>
    git remote add origin <url>
    git push -u origin main

Start with README.md.
