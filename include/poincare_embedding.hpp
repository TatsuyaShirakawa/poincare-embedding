#ifndef POINCARE_EMBEDDING_HPP
#define POINCARE_EMBEDDING_HPP

#include <cassert>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <thread>

namespace poincare_disc{

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Initializer
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Initializer
  {
  public:
    using real = RealType;
  public:
    virtual real operator()() = 0;
  };

  template <class RealType>
  struct ZeroInitializer: public Initializer<RealType>
  {
  public:
    using real = RealType;
  public:
    real operator()() { return 0.; }
  };

  template <class RealType>
  struct UniformInitializer: public Initializer<RealType>
  {
  public:
    using real = RealType;
  private:
    std::default_random_engine engine_;
    std::uniform_real_distribution<real> dist_;
  public:
    UniformInitializer(const real a, const real b, const unsigned int seed = 0)
      : engine_(seed), dist_(a, b)
    {}
  public:
    real operator()() { return dist_(engine_); }
  };


  ///////////////////////////////////////////////////////////////////////////////////////////
  // Matrix, VectorView
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct VectorView
  {
  public:
    using real = RealType;
  public:
    VectorView(): data_(nullptr), dim_(0) {}
    VectorView(real* data, std::size_t dim): data_(data), dim_(dim) {}
  public:
    const std::size_t dim() const { return dim_; }
    const real operator[](const std::size_t i) const { return data_[i]; }
    real& operator[](const std::size_t i) { return data_[i]; }

    VectorView<real>& assign_(const real c, const VectorView<real>& v)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_[i] = c * v.data_[i];
      }
      return *this;
    }

    VectorView<real>& add_(const real c, const VectorView<real>& v)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_[i] += c * v.data_[i];
      }
      return *this;
    }

    real squared_sum() const { return this->dot(*this); }
    real dot(const VectorView& v) const
    { return std::inner_product(data_, data_ + dim_, v.data_, 0.); }

  private:
    std::size_t dim_;
    real *data_;
  };

  template <class RealType>
  struct Matrix
  {
  public:
    using real = RealType;
  public:
    Matrix(): m_(0), n_(0), data_(nullptr) {}

    template <class Initializer>
    Matrix(const std::size_t m, const std::size_t n, Initializer initializer): m_(), n_(), data_()
    { init(m, n, initializer); }

  public:

    template <class Initializer>
    void init(const std::size_t m, const std::size_t n, Initializer initializer)
    {
      m_ = m; n_ = n; data_ = std::shared_ptr<real>(new real[m*n], [](real *p) { delete[] p;});
      for(std::size_t i=0, I=m*n; i < I; ++i){
        data_.get()[i] = initializer();
      }
    }

    std::size_t nrow() const { return m_; }
    std::size_t ncol() const { return n_; }

    const VectorView<real> operator[](const std::size_t i) const
    { return VectorView<real>(data_.get() + n_ * i, n_); }

    VectorView<real> operator[](const std::size_t i)
    { return VectorView<real>(data_.get() + n_ * i, n_); }

    void zero_()
    {
      for(auto ptr = data_.get(), end = data_.get() + m_ * n_; ptr != end; ++ptr){
        *ptr = 0.;
      }
    }

  private:
    std::size_t m_, n_;
    std::shared_ptr<real> data_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Matrix, VectorView
  ///////////////////////////////////////////////////////////////////////////////////////////



  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Disc
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  RealType arcosh(const RealType x)
  {
    assert( x > 1 );
    return std::log(x + std::sqrt(x*x - 1)); 
  }

  template <class RealType>
  struct Distance
  {
  public:
    using real = RealType;
  public:
    Distance(): u_(), v_(), uu_(), vv_(), uv_(), alpha_(), beta_(), gamma_() {}
    real operator()(const VectorView<real>& u, const VectorView<real>& v)
    {
      u_ = u;
      v_ = v;
      uu_ = u_.squared_sum();
      vv_ = v_.squared_sum();
      uv_ = u_.dot(v_);
      alpha_ = 1 - uu_;
      beta_ = 1 - vv_;
      gamma_ = 1 + 2 * (uu_ - 2 * uv_ + vv_) / alpha_ / beta_;
      return arcosh<real>(gamma_);
    }

    void backward(VectorView<real>& grad_u, VectorView<real>& grad_v, real grad_output)
    {
      real c = grad_output;

      c *= 4 / beta_ / std::sqrt(gamma_ * gamma_ - 1);

      // adjust by metric^(-1)
      real cu = c * (1 - uu_) * (1 - uu_) / 4;
      real cv = c * (1 - vv_) * (1 - vv_) / 4;

      grad_u.assign_(cu * (vv_ - 2 * uv_ + 1) / alpha_ / alpha_, u_);
      grad_u.add_(-cu / alpha_, v_);

      grad_v.assign_(cv * (uu_ - 2 * uv_ + 1) / alpha_ / alpha_, v_);
      grad_v.add_(-cv / alpha_, u_);
    }

  private:
    VectorView<real> u_, v_;
    real uu_, vv_, uv_, alpha_, beta_, gamma_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Negative Sampler
  ///////////////////////////////////////////////////////////////////////////////////////////

  // TODO: reject pairs which appear in dataset
  struct UniformNegativeSampler
  {
  public:

    template <class InputIt>
    UniformNegativeSampler(InputIt first, InputIt last, unsigned int seed)
      : engine_(seed), dist_(first, last)
    {}

    std::size_t operator()()
    { return dist_(engine_); }

  private:
    std::default_random_engine engine_;
    std::discrete_distribution<std::size_t> dist_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Loss Function
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct CrossEntropyLoss
  {
  public:
    using real = RealType;
  public:
    real operator()(const real dist) const
    {}


  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Utilities
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class KeyType>
  struct Dictionary
  {
  public:
    using key_type = KeyType;
  public:
    Dictionary(): hash_(), keys_(), counts_() {}
    std::size_t size() const { return hash_.size(); }
    bool find(const key_type& key) const { return hash_.find(key) != hash_.end(); }
    std::size_t put(const key_type& key)
    {
      auto itr = hash_.find(key);
      if(itr == hash_.end()){
        std::size_t n = size();
        hash_.insert(std::make_pair(key, n));
        keys_.push_back(key);
        counts_.push_back(1);
        return n;
      }
      std::size_t n = itr->second;
      ++counts_[n];
      return n;
    }
    std::size_t get_hash(const key_type& key) const {
      return hash_.find(key)->second;
    }
    key_type get_key(const std::size_t i) const { return keys_[i]; }
    std::size_t get_count(const std::size_t i) const { return counts_[i]; }

    const std::vector<std::size_t>& counts() const { return counts_; }
  private:
    std::unordered_map<key_type, std::size_t> hash_;
    std::vector<key_type> keys_;
    std::vector<std::size_t> counts_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Embedding
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Config
  {
    using real = RealType;
    std::size_t dim = 5; // dimension
    std::size_t seed = 0; // seed
    UniformInitializer<real> initializer; // embedding initializer
    std::size_t num_threads = 1;
    std::size_t neg_size = 10;
    std::size_t max_epoch = 1;
    char delim = ',';
    real lr0 = 0.0001; // learning rate
  };

  inline bool read_data(std::vector<std::pair<std::size_t, std::size_t> >& data,
                  Dictionary<std::string>& dict,
                  const std::string& filename,
                  const char delim)
  {
    std::ifstream fin(filename.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot read file: " << filename << std::endl;
      return false;
    }

    // TODO incremental read
    fin.seekg(0, fin.end);
    int length = fin.tellg();
    fin.seekg(0, fin.beg);
    char* buffer = new char[length+1];
    fin.read(buffer, length);
    buffer[length] = '\0';
    fin.close();

    std::size_t pos_start = 0, pos_end = 0;
    std::size_t line_no = 0;
    while(pos_start < length){

      assert(pos_start == pos_end);

      // skip until \n
      while(pos_end < length && buffer[pos_end] != '\n'){
        ++pos_end;
      }

      // line found
      ++line_no;
      if(pos_start == pos_end){ // empty line
        std::cerr << "empty line found in " << filename << ", line no: " << line_no << std::endl;
        delete [] buffer;
        return false;
      }
      buffer[pos_end] = '\0';

      // process non empty line
      char* line = buffer + pos_start;
      std::size_t line_length = pos_end - pos_start;
      std::size_t pos_field_start = 0, pos_field_end = 0;

      // find delim
      while(pos_field_end < line_length && line[pos_field_end] != delim){
        ++pos_field_end;
      }

      if(pos_field_end == pos_end){
        std::cerr << "no delimiter found: " << line << std::endl;
        delete [] buffer;
        return false;
      }

      // entry fields
      line[pos_field_end] = '\0';
      std::size_t n1 = dict.put(line);
      std::size_t n2 = dict.put(line + pos_field_end + 1);
      data.push_back(std::make_pair(n1, n2));

      // next line
      pos_start = (++pos_end);
    }

    delete [] buffer;
    return true;
  }

  template <class RealType, class DataItr>
  bool train_thread(Matrix<RealType>& embeddings,
                    const std::vector<std::size_t>& counts,
                    DataItr beg, DataItr end,
                    const Config<RealType>& config,
                    const unsigned int seed)
  {
    using real = RealType;

    // construct negative sampler
    UniformNegativeSampler negative_sampler(counts.begin(), counts.end(), seed);

    // data, gradients, distances
    std::vector<std::size_t> left_indices(1 + config.neg_size), right_indices(1 + config.neg_size);
    Matrix<real> left_grads(1, config.dim(), ZeroInitializer<real>()); // u
    Matrix<real> right_grads(1 + config.neg_size, config.dim(), ZeroInitializer<real>()); // v, v', ...
    std::vector<Distance<real> > dists(1 + config.neg_size);
    std::vector<real> exp_neg_dist_values(1 + config.neg_size);
    // start training
    auto itr = beg;
    while(itr != end){

      // // zero init gradients
      // left_grads.zero_();
      // right_grads.zero_();

      // store samples
      auto i = left_indices[0] = itr->first;
      auto j = right_indices[0] = itr->second;
      exp_neg_dist_values[0] = std::exp(-dists[0](embeddings[i], embeddings[j]));
      for(std::size_t k = 0; k < config.neg_size; ++k){
        auto i = left_indices[k + 1] = itr->first;
        auto j = right_indices[k + 1] = negative_sampler();
        exp_neg_dist_values[k + 1] = std::exp(-dists[k + 1](embeddings[i], embeddings[j]));
      }

      // compute gradient
      // grads for 0
      dists[0].backward(left_grads[0], right_grads[0], 1.);
      // grads for 1, 2, ...
      // at first, compute the grad input
      real Z = 0.;
      for(std::size_t k = 0; k < config.neg_size; ++k){
        Z += exp_neg_dist_values[k + 1];
      }
      for(std::size_t k = 0; k < config.neg_size; ++k){
        dists[k + 1].backward(left_grads[k+1], right_grads[k+1], -exp_neg_dist_values[k+1]/Z);
      }

      // update
      for(std::size_t k = 0; k < 1 + config.neg_size; ++k){
        auto i = left_indices[k], j = right_indices[k];
        embeddings[i].add_(-config.lr0, left_grads[k]);
        embeddings[j].add_(-config.lr0, right_grads[k]);
      }

      // next iteration
      ++itr;
    }
  }

  template <class RealType>
  bool poincare_embedding(Matrix<RealType>& embeddings,
                          const std::string& filename,
                          const Config<RealType>& config)
  {
    using real = RealType;

    std::default_random_engine engine(config.seed);

    // read file and construct negative sampler
    std::vector<std::pair<std::size_t, std::size_t> > data;
    Dictionary<std::string> dict;

    bool ret = read_data(data, dict, filename, config.delim);

    std::size_t data_size = data.size();
    std::cout << "data size: " << data_size << std::endl;

    std::ifstream fin(filename.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot read file: " << filename << std::endl;
      return false;
    }

    embeddings.init(dict.size(), config.dim, config.initializer);

    // fit
    std::vector<std::pair<std::size_t, std::size_t> > fake_pairs(config.neg_size);
    for(std::size_t epoch = 0; epoch < config.max_epoch; ++epoch){
      std::cout << "epoch " << epoch+1 << "/" << config.max_epoch << " start" << std::endl;
      std::cout << "random shuffle data" << std::endl;
      std::random_shuffle(data.begin(), data.end());

      if(config.num_threas > 1){
        // multi thread
        std::size_t data_size_per_thread = data_size / config.num_threads;
        std::cout << "start multi thread training" << std::endl;
        std::cout << "num_threads = " << config.num_threads << std::endl;
        std::cout << "data size = " << data_size_per_thread << "/thread" << std::endl;

        std::vector<std::thread> threads;
        for(std::size_t i = 0; i < config.num_threads; ++i){
          auto beg = data.begin() + data_size_per_thread * i;
          auto end = data.begin() + std::min(data_size_per_thread * (i+1), data_size);
          unsigned int thread_seed = engine();
          threads.push_back(std::thread( [=]{ train_thread(embeddings, dict.counts(),
                                                           beg, end,
                                                           config, thread_seed); }  ));
        }
        for(auto& th : threads){
          th.join();
        }
      }else{
        // single thread
        std::cout << "start single thread training" << std::endl;
        std::cout << "data size = " << data_size << std::endl;
        const unsigned int thread_seed = engine();
        train_thread(embeddings, dict.counts(), data.begin(), data.end(), config, thread_seed);
      }

    }

    return true;
  }

}

#endif
