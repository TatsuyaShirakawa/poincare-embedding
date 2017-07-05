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

struct Arguments
{
  std::string data_file;
  std::string result_embedding_file = "embeddings.csv";
  unsigned int seed = 0;
  std::size_t num_threads = 1;
  std::size_t neg_size = 10;
  std::size_t max_epoch = 1;
  std::size_t dim = 50;
  real lr0 = 0.001;
};

Arguments parse_args(int narg, char** argv)
{
  Arguments result;
  std::size_t arg_count = 0;
  std::string program_name = argv[0];
  for(int i = 1; i < narg; ++i){
    std::string arg(argv[i]);
    if(arg == "-s" || arg == "--seed"){
      arg = argv[++i];
      int n = std::stol(arg);
      if( n < 0 ){ goto HELP; }
      result.seed = static_cast<unsigned int>(n);
      continue;
    }else if(arg == "-t" || arg == "--num_thread"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.num_threads = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-n" || arg == "--neg_size"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.neg_size = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-e" || arg == "--max_epoch"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.max_epoch = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-d" || arg == "--dim"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.dim = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-l" || arg == "--learning_rate"){
      arg = argv[++i];
      double x = std::stod(arg);
      if( x <= 0 ){ goto HELP; }
      result.lr0 = static_cast<real>(x);
      continue;
    }else if(arg == "-h" || arg == "--help"){
      goto HELP;
    }

    if(arg_count == 0){
      result.data_file = arg;
      ++arg_count;
      continue;
    }else if(arg_count == 1){
      result.result_embedding_file = arg;
      ++arg_count;
      continue;
    }

    std::cerr << "invalid argument: " << arg << std::endl;
    goto HELP;
  }

  if(arg_count == 0){
    std::cerr << "missing argments" << std::endl;
    goto HELP;
  }else if(arg_count > 2){
    std::cerr << "too many argments" << std::endl;
    goto HELP;
  }

  return result;

 HELP:
  std::cerr <<
    program_name << " data_file (result_embedding_file) [options...]\n"
    "\n"
    "    data_file            : string    input data tsv file. each line contains pair of items\n"
    "    result_embeddng_file : string    result file into which resulting embeddings are written\n"
    "    -s, --seed           : int >= 0   random seed\n"
    "    -t, --num_threads    : int > 0   number of threads\n"
    "    -n, --neg_size       : int > 0   negativ sample size\n"
    "    -e, --max_epoch      : int > 0   maximum training epochs\n"
    "    -d, --dim            : int > 0   dimension of embeddings\n"
    "    -l, --learning_rate  : float > 0 initial learning rate\n"
            << std::endl;
  exit(0);
}



int main(int narg, char** argv)
{
  Arguments args = parse_args(narg, argv);


  std::string data_file = args.data_file;
  std::string result_embedding_file = args.result_embedding_file;

  Matrix<real> embeddings;
  Dictionary<std::string> dict;
  Config<real> config;
  config.seed = args.seed;
  config.num_threads = args.num_threads;
  config.neg_size = args.neg_size;
  config.max_epoch = args.max_epoch;
  config.dim = args.dim;
  config.lr0 = args.lr0;

  std::cout << "settings:" << "\n"
            << "  " << "data_file             : " << data_file << "\n"
            << "  " << "result_embedding_file : " << result_embedding_file << "\n"
            << "  " << "seed                  : " << config.seed << "\n"
            << "  " << "num_threads           : " << config.num_threads << "\n"
            << "  " << "neg_size              : " << config.neg_size << "\n"
            << "  " << "max_epoch             : " << config.max_epoch << "\n"
            << "  " << "dim                   : " << config.dim << "\n"
            << "  " << "lr0                   : " << config.lr0 << "\n"
            << std::endl;

  std::cout << "start training" << std::endl;
  bool ret = poincare_embedding<real>(embeddings, dict, data_file, config);

  std::cout << "save to " << result_embedding_file << std::endl;
  save(result_embedding_file, embeddings, dict);

  return 0;
}
