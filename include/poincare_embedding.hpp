#ifndef POINCARE_EMBEDDING_HPP
#define POINCARE_EMBEDDING_HPP

#include <cassert>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iomanip>

#define LEFT_SAMPLING 0
#define RIGHT_SAMPLING 1
#define BOTH_SAMPLING 2
#define SAMPLING_STRATEGY 1

namespace poincare_disc{

  constexpr float EPS = 0.00001;

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
  // Vector, Matrix
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Vector
  {
  public:
    using real = RealType;
  public:
    Vector(): data_(nullptr), dim_(0) {}
    Vector(std::shared_ptr<real> data, std::size_t dim): data_(data), dim_(dim) {}
    Vector(const Vector<real>& v): data_(v.data_), dim_(v.dim_) {}
    Vector<real>& operator=(const Vector<real>& v)
    {
      data_ = v.data_; dim_ = v.dim_;
      return *this;
    }
  public:
    const std::size_t dim() const { return dim_; }
    const real operator[](const std::size_t i) const { return data_.get()[i]; }
    real& operator[](const std::size_t i) { return data_.get()[i]; }

    Vector<real>& assign_(const real c, const Vector<real>& v)
    {
      if(dim_ != v.dim_){
        dim_ = v.dim_;
        data_ = std::shared_ptr<real>(new real[v.dim_]);
      }
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] = c * v.data_.get()[i];
      }
      return *this;
    }

    Vector<real>& zero_()
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] = 0;
      }
      return *this;
    }

    Vector<real>& add_(const real c, const Vector<real>& v)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] += c * v.data_.get()[i];
      }
      return *this;
    }

    Vector<real>& add_clip_(const real c, const Vector<real>& v, const real thresh=1.0-EPS)
    {
      real uu = this->squared_sum(), uv = this->dot(v), vv = v.squared_sum();
      real C = uu + 2*c*uv + c*c*vv; // resulting norm
      real scale = 1.0;
      if(C > thresh * thresh){
        scale = thresh / sqrt(C);
      }
      assert( 0 < scale && scale <= 1. );
      if(scale == 1.){
        for(int i = 0, I = dim(); i < I; ++i){
          data_.get()[i] += c * v.data_.get()[i];
        }
      }else{
        for(int i = 0, I = dim(); i < I; ++i){
          data_.get()[i] = (data_.get()[i] + c * v.data_.get()[i]) * scale;
        }
      }
      assert(this->squared_sum() <= (thresh + EPS) * (thresh+EPS));
      return *this;
    }

    Vector<real>& mult_(const real c)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] *= c;
      }
      return *this;
    }

    real squared_sum() const { return this->dot(*this); }

    real dot(const Vector& v) const
    { return std::inner_product(data_.get(), data_.get() + dim_, v.data_.get(), 0.); }


  private:
    std::size_t dim_;
    std::shared_ptr<real> data_;
  };


  template <class RealType>
  std::ostream& operator<<(std::ostream& out, const Vector<RealType>& v)
  {
    if(v.dim() < 5){
      out << "[";
      for(int i = 0; i < v.dim(); ++i){
        if(i > 0){ out << ", ";}
        out << v[i];
      }
      out << "]";
    }else{
      out << "[";
      out << v[0] << ", " << v[1] << ", ..., " << v[v.dim()-1];
      out << "]";
    }
    return out;
  }

  template <class RealType>
  struct Matrix
  {
  public:
    using real = RealType;
  public:
    Matrix(): m_(0), n_(0), rows_() {}

    template <class Initializer>
    Matrix(const std::size_t m, const std::size_t n, Initializer initializer): m_(), n_(), rows_()
    { init(m, n, initializer); }

    Matrix(const Matrix<real>& mat): m_(mat.m_), n_(mat.n_), rows_(mat.rows_) {}

    Matrix<real>& operator=(const Matrix<real>& mat)
    {
      m_ = mat.m_;
      n_ = mat.n_;
      rows_ = mat.rows_;
      return *this;
    }

  public:

    template <class Initializer>
    void init(const std::size_t m, const std::size_t n, Initializer initializer)
    {
      m_ = m; n_ = n; rows_ = std::vector<Vector<real> >(m);
      for(std::size_t i = 0; i < m; ++i){
        rows_[i] = Vector<real>(std::shared_ptr<real>(new real[n]), n);
        for(std::size_t j = 0; j < n; ++j){
          rows_[i][j] = initializer();
        }
      }
    }

    std::size_t nrow() const { return m_; }
    std::size_t ncol() const { return n_; }

    const Vector<real>& operator[](const std::size_t i) const
    { return rows_[i]; }

    Vector<real>& operator[](const std::size_t i)
    { return rows_[i]; }

    Matrix<real>& zero_()
    {
      for(std::size_t i = 0; i < m_; ++i){
        rows_[i].zero_();
      }
      return *this;
    }

  private:
    std::size_t m_, n_;
    std::vector<Vector<real> > rows_;
  };


  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Disc
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  RealType arcosh(const RealType x)
  {
    assert( x >= 1 );
    return std::log(x + std::sqrt(x*x - 1)); 
  }

  template <class RealType>
  struct Distance
  {
  public:
    using real = RealType;
  public:
    Distance(): u_(), v_(), uu_(), vv_(), uv_(), alpha_(), beta_(), gamma_() {}
    real operator()(const Vector<real>& u, const Vector<real>& v)
    {
      u_ = u;
      v_ = v;
      uu_ = u_.squared_sum();
      vv_ = v_.squared_sum();
      uv_ = u_.dot(v_);
      alpha_ = 1 - uu_;
      if(alpha_ <= 0){ alpha_ = EPS; } // TODO: ensure 0 <= uu_ <= 1-EPS;
      // if(!(alpha_ > 0)){ std::cout << "uu_: " << uu_ << ", alpha_: " << alpha_ << std::endl; }
      // assert(alpha_ > 0);
      beta_ = 1 - vv_;
      if(beta_ <= 0){ beta_ = EPS; } // TODO: ensure 0 <= vv_ <= 1-EPS;
      // if(!(beta_ > 0)){ std::cout << "vv_: " << vv_ << ", beta_: " << beta_ << std::endl; }
      // assert(beta_ > 0);
      gamma_ = 1 + 2 * (uu_ - 2 * uv_ + vv_) / alpha_ / beta_;
      if(gamma_ < 1.){ gamma_ = 1.; } // for nemerical error
      assert(gamma_ >= 1);
      return arcosh<real>(gamma_);
    }

    void backward(Vector<real>& grad_u, Vector<real>& grad_v, real grad_output)
    {
      real c = grad_output;
      if(gamma_ == 1){
        grad_u.zero_();
        grad_v.zero_();
        return;
      }

      c  *= 4 / std::sqrt(gamma_ * gamma_ - 1) / alpha_ / beta_;

      // grad for u
      real cu = c * alpha_ * alpha_ / 4;
      real cv = c * beta_ * beta_  / 4;

      grad_u.assign_(cu * (vv_ - 2 * uv_ + 1) / alpha_, u_);
      grad_u.add_(-cu, v_);

      grad_v.assign_(cv * (uu_ - 2 * uv_ + 1) / beta_, v_);
      grad_v.add_(-cv, u_);
    }

  private:
    Vector<real> u_, v_;
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
  // Optimization
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct LinearLearningRate
  {
  public:
    using real = RealType;
  public:
    LinearLearningRate(const real lr_init, const real lr_final, const std::size_t total_iter)
      :lr_init_(lr_init), lr_final_(lr_final), current_iter_(0), total_iter_(total_iter)
    {}
  public:
    void update(){ ++current_iter_;}
    real operator()() const
    {
      real r = static_cast<real>(static_cast<double>(current_iter_) / total_iter_);
      assert( 0 <= r && r <= 1);
      return (1-r) * lr_init_ + r * lr_final_;
    }
  public:
    real lr_init_;
    real lr_final_;
    std::size_t current_iter_;
    std::size_t total_iter_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Embedding
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Config
  {
    using real = RealType;
    std::size_t dim = 5; // dimension
    unsigned int seed = 0; // seed
    UniformInitializer<real> initializer = UniformInitializer<real>(-0.0001, 0.0001); // embedding initializer
    std::size_t num_threads = 1;
    std::size_t neg_size = 10;
    std::size_t max_epoch = 1;
    char delim = '\t';
    real lr0 = 0.01; // learning rate
    real lr1 = 0.0001; // learning rate
  };

  template <class RealType>
  void clip(Vector<RealType>& v, const RealType& thresh = 1-EPS)
  {
    RealType vv = v.squared_sum();
    if(vv >= thresh*thresh){
      v.mult_(thresh / std::sqrt(vv));
    }
  }

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
                    LinearLearningRate<RealType>& lr,
                    const std::size_t thread_no,
                    const unsigned int seed)
  {
    using real = RealType;

    // clip
    for(std::size_t i = 0, I = embeddings.nrow(); i < I; ++i){
      clip(embeddings[i]);
    }

    // construct negative sampler
    // TODO: negative sampler can be shared through threads
    UniformNegativeSampler negative_sampler(counts.begin(), counts.end(), seed);

    // data, gradients, distances
    std::vector<std::size_t> left_indices(1 + config.neg_size), right_indices(1 + config.neg_size);
    Matrix<real> left_grads(1 + config.neg_size, config.dim, ZeroInitializer<real>()); // u
    Matrix<real> right_grads(1 + config.neg_size, config.dim, ZeroInitializer<real>()); // v, v', ...
    std::vector<Distance<real> > dists(1 + config.neg_size);
    std::vector<real> exp_neg_dist_values(1 + config.neg_size);
    // start training
    auto itr = beg;
    std::size_t itr_count = 0, total_itr = std::distance(beg, end);
    auto tick = std::chrono::system_clock::now();
    auto start_time = tick;
    constexpr std::size_t progress_interval = 10000;
    double avg_loss = 0;
	    double cum_loss = 0;
    // if(thread_no == 0){
    //   std::cout << embeddings[0] << std::endl;
    //   std::cout << embeddings[1000] << std::endl;
    //   std::cout << embeddings[2000] << std::endl;
    // }
    while(itr != end){
      if(thread_no == 0 && itr_count % progress_interval == 0){
        auto tack = std::chrono::system_clock::now();
        auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
        tick = tack;
        double percent = (100.0 * itr_count) / total_itr;
        cum_loss += avg_loss;
        avg_loss /= progress_interval;
        std::cout << "\r"
                  <<std::setw(5) << std::fixed << std::setprecision(2) << percent << " %"
                  << "    " << config.num_threads * progress_interval*1000./millisec << " itr/sec"
                  << "    " << "loss: " << avg_loss
                  << std::flush;

        avg_loss = 0;
      }
      ++itr_count;
      // // zero init gradients
      // left_grads.zero_();
      // right_grads.zero_();

      // store samples
      auto i = left_indices[0] = itr->first;
      auto j = right_indices[0] = itr->second;

      exp_neg_dist_values[0] = std::exp(-dists[0](embeddings[i], embeddings[j]));
      for(std::size_t k = 0; k < config.neg_size; ++k){
#if SAMPLING_STRATEGY == LEFT_SAMPLING
        auto i = left_indices[k + 1] = negative_sampler();
        auto j = right_indices[k + 1] = itr->second;
#elif SAMPLING_STRATEGY == RIGHT_SAMPLING
        auto i = left_indices[k + 1] = itr->first;
        auto j = right_indices[k + 1] = negative_sampler();
#elif SAMPLING_STRATEGY == BOTH_SAMPLING
        auto i = left_indices[k + 1] = negative_sampler();
        auto j = right_indices[k + 1] = negative_sampler();
#endif
        exp_neg_dist_values[k + 1] = std::exp(-dists[k + 1](embeddings[i], embeddings[j]));
      }

      // compute gradient
      // grads for 1, 2, ...
      // at first, compute the grad input
      real Z = exp_neg_dist_values[0];
      for(std::size_t k = 0; k < config.neg_size; ++k){
        Z += exp_neg_dist_values[k + 1];
      }
      for(std::size_t k = 0; k < config.neg_size; ++k){
        dists[k + 1].backward(left_grads[k+1], right_grads[k+1], -exp_neg_dist_values[k+1]/Z);
      }
      // grads for 0
      dists[0].backward(left_grads[0], right_grads[0], 1 - exp_neg_dist_values[0]/Z);

      // add loss
      {
        avg_loss -= std::log(exp_neg_dist_values[0]);
        avg_loss += std::log(Z);
      }


      // update
      for(std::size_t k = 0; k < 1 + config.neg_size; ++k){
        auto i = left_indices[k], j = right_indices[k];
        embeddings[i].add_clip_(-lr(), left_grads[k]);
        embeddings[j].add_clip_(-lr(), right_grads[k]);
      }

      lr.update();

      // // clip
      // for(std::size_t k = 0; k < 1 + config.neg_size; ++k){
      //   auto i = left_indices[k], j = right_indices[k];
      //   clip(embeddings[i]);
      //   clip(embeddings[j]);
      // }

      // next iteration
      ++itr;
    }
    if(thread_no == 0){
      cum_loss += avg_loss;
      auto tack = std::chrono::system_clock::now();
      auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-start_time).count();
        std::cout << "\r"
                  <<std::setw(5) << std::fixed << std::setprecision(2) << 100 << " %"
                  << "    " << config.num_threads * total_itr * 1000./millisec << " itr/sec"
                  << "    " << "loss: " << cum_loss / total_itr
                  << std::endl;
    }
    return true;
  }

  template <class RealType>
  bool poincare_embedding(Matrix<RealType>& embeddings,
                          Dictionary<std::string>& dict,
                          const std::string& filename,
                          const Config<RealType>& config)
  {
    using real = RealType;

    std::default_random_engine engine(config.seed);

    // read file and construct negative sampler
    std::vector<std::pair<std::size_t, std::size_t> > data;
    std::cout << "read " << filename << std::endl;
    bool ret = read_data(data, dict, filename, config.delim);

    std::size_t data_size = data.size();
    std::cout << "data size: " << data_size << std::endl;

    std::ifstream fin(filename.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot read file: " << filename << std::endl;
      return false;
    }

    embeddings.init(dict.size(), config.dim, config.initializer);

    std::cout << "embedding size: " << embeddings.nrow() << " x " << embeddings.ncol() << std::endl;

    // fit
    LinearLearningRate<real> lr(config.lr0, config.lr1, data_size * config.max_epoch);
    std::vector<std::pair<std::size_t, std::size_t> > fake_pairs(config.neg_size);
    std::cout << "num_threads = " << config.num_threads << std::endl;
    std::size_t data_size_per_thread = data_size / config.num_threads;
    std::cout << "data size = " << data_size_per_thread << "/thread" << std::endl;

    for(std::size_t epoch = 0; epoch < config.max_epoch; ++epoch){
      std::cout << "epoch " << epoch+1 << "/" << config.max_epoch << " start" << std::endl;
      // std::cout << "random shuffle data" << std::endl;
      std::random_shuffle(data.begin(), data.end());

      if(config.num_threads > 1){
        // multi thread

        std::vector<std::thread> threads;
        for(std::size_t i = 0; i < config.num_threads; ++i){
          auto beg = data.begin() + data_size_per_thread * i;
          auto end = data.begin() + std::min(data_size_per_thread * (i+1), data_size);
          unsigned int thread_seed = engine();
          const auto& counts = dict.counts();
          threads.push_back(std::thread( [=, &embeddings, &counts, &lr]{ train_thread(embeddings, counts,
                                                                                      beg, end,
                                                                                      config, lr,
                                                                                      i, thread_seed); }  ));
        }
        for(auto& th : threads){
          th.join();
        }
      }else{
        // single thread
        const unsigned int thread_seed = engine();
        train_thread(embeddings, dict.counts(), data.begin(), data.end(), config, lr, 0, thread_seed);
      }

    }

    return true;
  }

}

#endif
