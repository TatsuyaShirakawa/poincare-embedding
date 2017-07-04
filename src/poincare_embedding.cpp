#include <iostream>
#include <string>
#include <fstream>
#include "poincare_embedding.hpp"

using namespace poincare_disc;
using real = double;

void save(const std::string& filename,
          const Matrix<real>& embeddings,
          const Dictionary<std::string>& dict)
{
  std::ofstream fout(filename.c_str());
  if(!fout || !fout.good()){
    std::cerr << "file cannot be open: " << filename << std::endl;
  }
  for(std::size_t i = 0, I = dict.size(); i < I; ++i){
    fout << dict.get_key(i);
    for(std::size_t k = 0, K = embeddings.ncol(); k < K; ++k){
      fout << "\t" << embeddings[i][k];
    }
    fout << "\n";
  }
}

int main()
{
  std::string filename = "./pair.full.text8.csv";
  std::string resultfile = "./embeddings.tsv";

  Matrix<real> embeddings;
  Dictionary<std::string> dict;
  Config<real> config;
  config.num_threads = 4;

  std::cout << "start training" << std::endl;
  bool ret = poincare_embedding<real>(embeddings, dict, filename, config);

  std::cout << "save to " << resultfile << std::endl;
  save(resultfile, embeddings, dict);

  return 0;
}
