#ifndef __GRAPH_AUX_H__
#define __GRAPH_AUX_H__

#include "critter.h"
#include <ctf.hpp>
#include <float.h>
#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
typedef int wht;

uint64_t norm_graph(uint64_t *ed, uint64_t ned);
uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge);
uint64_t read_graph_mpiio(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led);
void processedges(char **led, uint64_t ned, int myid, int ntasks, uint64_t **edge, uint64_t start, bool eweights, wht ** vals);
// uint64_t read_metis(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led, int * n, uint64_t * start, bool * eweights, wht ** vals);
uint64_t read_metis(int myid, int ntask, const char *fpath, std::vector<std::pair<uint64_t, uint64_t> > &edges, int * n, bool * e_weights, std::vector<wht> &eweights);
#endif

