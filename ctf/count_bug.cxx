/*
 *  Demonstrates the bug when computing the Scalar value count
 *  mpirun -np 3 ./count_bug 8 
 *  calculates the count as 5 (instead of 4)
 */
#include <ctf.hpp>

using namespace CTF;

void init_vector(Vector<int>* p, int offset)
{
  int64_t npairs;
  Pair<int> * loc_pairs;
  p->read_local(&npairs, &loc_pairs);
  for (int64_t i = 0; i < npairs; i++){
    if (loc_pairs[i].k % offset == 0) {
      loc_pairs[i].d = loc_pairs[i].k;
    }
  }
  p->write(npairs, loc_pairs);
  delete [] loc_pairs;
}

int main(int argc, char **argv)
{
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  
  int vec_size;
  if (argc < 2) {
    std::cout << "Usage ./count_bug <vec_size>" << std::endl;
    return -1;
  }
  vec_size = atoi(argv[1]);
  
  auto p = new Vector<int>(vec_size, *w);
  init_vector(p, 1);
  
  auto q = new Vector<int>(vec_size, *w);
  init_vector(q, 2);
  
  Scalar<int> count(*w);
  count[""] = Function<int,int,int>([](int a, int b){ return (int64_t)(a==b); })((*p)["i"], q->operator[]("i"));

  if (w->rank == 0) std::cout << "Printing q vector" << std::endl;
  q->print();
  if (w->rank == 0) std::cout << "Count: " << count << endl;
  return 0;

}
