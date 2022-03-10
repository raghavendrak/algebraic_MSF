# Minimum Spanning Forest

## Compiling
* In `ctf/generator`, run `make -f Makefile.BFS.mpi`.
* Download this [release of CTF](https://github.com/raghavendrak/ctf/releases/tag/v1.5.6).
* In `ctf`, run `make` to create the `test_mst` executable.

## Strong scaling
Run with a R-MAT graph with 2^S vertices and average edge degree E. 
`./test\_mst -S 6 -E 8`

## Weak scaling
Run with a uniform graph with n vertices and 0.005\*n^2 edges.
`./test_mst -n 1000 -sp 0.005`

## Snap dataset

## Matrix market

## From file

## Command line arguments
* `shortcut3 X`: trigger complete shortcutting if global number of vertices with changed parent is less than X. We used 131072.
* `-as 1`: pairwise implementation of hook step.

## Citation
If you use this implementation, please cite this [paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611977141.7).
