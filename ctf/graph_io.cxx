#include "graph_aux.h"

static void *Realloc(void *ptr, size_t sz) {

	void *lp;

	lp = (void *) realloc(ptr, sz);
	if (!lp && sz) {
		fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

	FILE *fp = NULL;
	fp = fopen(path, mode);
	if (!fp) {
		fprintf(stderr, "Cannot open file %s...\n", path);
		exit(EXIT_FAILURE);
	}
	return fp;
}

static uint64_t getFsize(FILE *fp) {

	int64_t rv;
	uint64_t size = 0;

	rv = fseek(fp, 0, SEEK_END);
	if (rv != 0) {
		fprintf(stderr, "SEEK END FAILED\n");
		if (ferror(fp)) fprintf(stderr, "FERROR SET\n");
		exit(EXIT_FAILURE);
	}

	size = ftell(fp);
	rv = fseek(fp, 0, SEEK_SET);

	if (rv != 0) {
		fprintf(stderr, "SEEK SET FAILED\n");
		exit(EXIT_FAILURE);
	}

	return size;
}

void processedges(char **led, uint64_t ned, int myid, int ntasks, uint64_t **edges, uint64_t start, bool eweights, wht ** vals) {
	uint64_t *ed=(uint64_t *)malloc(ned*sizeof(uint64_t)*2);
  /*
	for (int64_t i=0; i<ned; i++) {
		uint64_t a, b;
		sscanf(led[i],"%lu %lu", &a, &b);
		ed[i*2]  = a;
		ed[i*2+1]= b;
	}
  */
  for (int64_t i=0; i<ned; ++i) {
    int offset = 0;
    int itr = 0;
    while (true) { // FIXME: bad
      int n;
      uint64_t a;
      int cnt = sscanf(led[i]+offset, "%lu %n", &a, &n);
      offset += n;
      if (cnt != 1) break;
      ++itr;
      if (itr < start + 1) // skip lines until start
        continue;
      ed[i*2]   = i * ntasks + (i % ntasks) + 1; // vertices indexed at 1
      ed[i*2+1] = a;
      printf("edge: src: %lu, dest: %lu\n", ed[i*2], ed[i*2+1]);
      if (eweights) {
        cnt = sscanf(led[i]+offset, "%lu %n", &a, &n); // FIXME: update weights
        offset += n;
        if (cnt != 1) break;
        (*vals)[i] = a;
        printf("weight: %lu\n", (*vals)[i]);
        printf("\n");
      }
    }
  }

	*edges = ed;
}

// FIXME: how to track file numbers? a line may be parititioned to multiple processes
// if can track file numbers, do prefix sum to find start on this process
uint64_t read_graph_mpiio(int myid, int ntask, const char *fpath, uint64_t **edge, char ***led){ 
	MPI_File fh;
	MPI_Offset filesize;
	MPI_Offset localsize;
	MPI_Offset start,end;
	MPI_Status status;
	char *chunk = NULL;
	int MPI_RESULT = 0;
	int overlap = 100; // define
	int64_t ned = 0;
	int64_t i = 0;

	MPI_RESULT = MPI_File_open(MPI_COMM_WORLD,fpath, MPI_MODE_RDONLY, MPI_INFO_NULL,&fh);

	/* Get the size of file */
	MPI_File_get_size(fh, &filesize); //return in bytes 
  // FIXME: skewed to give lower processor counts more edges, since
  //        smaller node counts contain fewer characters
	localsize = filesize/ntask;
	start = myid * localsize;
	end = start + localsize;
	end +=overlap;
	
	if (myid  == ntask-1) end = filesize;
	localsize = end - start; //OK

	chunk = (char*)malloc( (localsize + 1)*sizeof(char)); 
	MPI_File_read_at_all(fh, start, chunk, localsize, MPI_CHAR, &status);
	chunk[localsize] = '\0';

	int64_t locstart=0, locend=localsize;
	if (myid != 0) {
		while(chunk[locstart] != '\n') locstart++;
		locstart++;
	}
	if (myid != ntask-1) {
		locend-=overlap;
		while(chunk[locend] != '\n') locend++;
    locend++;
	}
	localsize = locend-locstart; //OK

	char *data = (char *)malloc((localsize+1)*sizeof(char));
	memcpy(data, &(chunk[locstart]), localsize);
	data[localsize] = '\0';
	free(chunk);
	
  //printf("[%d] local chunk = [%ld,%ld) / %ld\n", myid, start+locstart, start+locstart+localsize, filesize);
	for ( i=0; i<localsize; i++){
		if (data[i] == '\n') ned++;
	}
  //printf("[%d] ned= %ld\n",myid, ned);

	(*led) = (char **)malloc(ned*sizeof(char *));
	(*led)[0] = strtok(data,"\n");

	for ( i=1; i < ned; i++)
		(*led)[i] = strtok(NULL, "\n");

	MPI_File_close(&fh);

	return ned;
}

uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge) {
#define ALLOC_BLOCK     (2*1024)
#define MAX_LINE        1024
   
	uint64_t *ed=NULL;
	uint64_t i, j;
	uint64_t n, nmax;
	uint64_t size;
	int64_t  off1, off2;

	int64_t  rem;
	FILE     *fp;
	char     str[MAX_LINE];

	fp = Fopen(fpath, "r");
	size = getFsize(fp);
	rem = size % ntask;
	off1 = (size/ntask)* myid    + (( myid    > rem)?rem: myid);
	off2 = (size/ntask)*(myid+1) + (((myid+1) > rem)?rem:(myid+1));

	if (myid < (ntask-1)) {
		fseek(fp, off2, SEEK_SET);
		fgets(str, MAX_LINE, fp);
		off2 = ftell(fp);
	}
	fseek(fp, off1, SEEK_SET);
	if (myid > 0) {
		fgets(str, MAX_LINE, fp);
		off1 = ftell(fp);
	}

	n = 0;
	nmax = ALLOC_BLOCK; // must be even
	ed = (uint64_t *)malloc(nmax*sizeof(*ed));
	uint64_t lcounter = 0;
	uint64_t nedges = -1;
	int64_t comment_counter = 0;

	/* read edges from file */
	while (ftell(fp) < off2) {

		// Read the whole line
		fgets(str, MAX_LINE, fp);

		// Strip # from the beginning of the line
		if (strstr(str, "#") != NULL) {
			//fprintf(stdout, "\nreading line number %"PRIu64": %s\n", lcounter, str);
			if (strstr(str, "Nodes:")) {
				//sscanf(str, "# Nodes: %" PRIu64 " Edges: %" PRIu64 "\n", &i, &nedges);
                sscanf(str, "# Nodes: %lu Edges: %lu\n", &i, &nedges);
				//fprintf(stdout, "N=%"PRIu64" E=%"PRIu64"\n", i, nedges);
			}
			comment_counter++;
		} else if (str[0] != '\0') {
			lcounter ++;
			// Read edges
//			sscanf(str, "%"PRIu64" %"PRIu64"\n", &i, &j);
	        sscanf(str, "%lu %lu\n", &i, &j);

			if (n >= nmax) {
				nmax += ALLOC_BLOCK;
				ed = (uint64_t *)Realloc(ed, nmax*sizeof(*ed));
			}
			ed[n]   = i;
			ed[n+1] = j;
			n += 2;
		}
	}
	fclose(fp);

	// number of ints -> number of edges
//	*edge = mirror(ed, &n); for undirected graph
    n /=2;
    *edge = ed;
	return n;
#undef ALLOC_BLOCK
}

uint64_t read_metis(int myid, int ntask, const char *fpath, std::vector<std::pair<uint64_t, uint64_t> > &edges, int * n, bool * e_weights, std::vector<wht> &eweights) {
  
  std::string fpaths = std::string(fpath);
  std::ifstream gfile(fpaths);
  if (!gfile.is_open()) {
    printf("I am rank: %d, I was unable to open file: %s\n", myid, fpaths.c_str());
  }
  std::string line_s;
  while (std::getline(gfile, line_s)) {
    if (line_s[0] != '%') break;
  }
  
  // process line_s which contains n, m
  char * header = &line_s[0];
  uint64_t parms[4]; // header may contain at most 4 parameters
  int parm_num = 0;
  int offset = 0;
  for (; parm_num < 4; ++parm_num) {
    int len;
    uint64_t a;
    int cnt = sscanf(header+offset, "%lu %n", &a, &len);
    offset += len;
    if (cnt != 1) break;
    parms[parm_num] = a;
  }
  assert(parm_num >= 2);
  *n = parms[0];
  uint64_t m = parms[1]; 
  uint64_t fmt = 0;
  uint64_t ncon = 0;
  uint64_t vweights = 0;
  uint64_t vsizes = 0;
  if (parm_num >= 3) {
    fmt = parms[2];
    if (fmt >= 100) {
      *e_weights = fmt % 10;
      vweights = (fmt / 10) % 10;
      vsizes = (fmt / 100) % 10;
    } else if (fmt >= 10) {
      *e_weights = fmt % 10;
      vweights = (fmt / 10) % 10;
    } else {
      *e_weights = fmt % 10;  
    }
  }

  if (parm_num >= 4)
    ncon = parms[3];
  if (myid == 0) printf("header: vertices = %ld edges = %ld fmt = %ld ncon = %ld\n", *n, m, fmt, ncon);  
  
  uint64_t lineNo = 1;
  uint64_t t_num;
  int64_t t_num_count;
  int64_t skip_num = 0;
  int64_t n_edges = 0;
  if (vsizes) skip_num += 1;
  if (ncon) skip_num += ncon;
  while (std::getline(gfile, line_s)) {
    if (line_s[0] == '%') continue;
    if (lineNo % ntask == myid) {
      // printf("I am rank: %d, I read line : %s\n", myid, line_s.c_str());
      std::stringstream line_ss(line_s);
      t_num_count = 0;
      while (line_ss >> t_num) {
        t_num_count++;
        if (t_num_count <= skip_num) continue;
        // std::cout << t_num << std::endl;
        edges.push_back(std::make_pair(lineNo, t_num)); 
        // get the weight
        line_ss >> t_num;
        t_num_count++;
        eweights.push_back(t_num);
        n_edges++;
      }
      assert(t_num_count > skip_num);
    }
    lineNo++;
  }
  return n_edges;
}
