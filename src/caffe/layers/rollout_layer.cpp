#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "caffe/layers/rollout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RolloutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  num_output_ = this->layer_param_.rollout_param().num_output();
  has_header_ = this->layer_param_.rollout_param().has_header();
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  data_dim_ = bottom[0]->count() / bottom[0]->num();
  if (has_header_) {
    CHECK_EQ(num_output_+1, data_dim_) << "input data dimension should be equal to num_output + 1 (header)";
  } else {
    CHECK_EQ(num_output_, data_dim_) << "input data dimension should be equal to num_output";
  }
}

template <typename Dtype>
void RolloutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(),1,1,1);
}

template <typename Dtype>
void RolloutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  for (int n = 0; n < num; ++n) {
 
    int rand_index = abs(caffe_rng_rand());
    if (!has_header_) {
      top_data[n] = bottom_data[n * data_dim_ + rand_index % num_output_];
    } 
    else {
      int max_len = static_cast<int>(bottom_data[n * data_dim_]);
	  int choice = rand_index % max_len;
//	  std::cout<<"max len: "<<max_len<<"Choice: "<<choice<<std::endl;
      top_data[n] = bottom_data[n * data_dim_ + choice + 1];
    }
  }
}

INSTANTIATE_CLASS(RolloutLayer);
REGISTER_LAYER_CLASS(Rollout);

}  // namespace caffe
