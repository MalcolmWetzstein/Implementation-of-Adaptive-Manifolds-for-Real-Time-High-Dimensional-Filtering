#ifndef A10_H_PHUDVTKB
#define A10_H_PHUDVTKB

#include <Eigen/Dense>
#include "Image.h"
#include "basicImageManipulation.h"
#include "filtering.h"


// Write your declarations here, or extend the Makefile if you add source
// files
enum Sampler { nearest_neighbor, linear, bicubic, lanczos };
Image adaptive_manifold_filter(const Image& f, float sigma_s, float sigma_r, Image (*const phi)(const Image&, float, const vector<float>&),
    const Image& f_joint, bool adjust_outliers = true, vector<Image>* clusters_out = nullptr, vector<Image>* manifolds_out = nullptr, int tree_height = 0,
    const vector<float>& phi_params = {}, int num_pca_iter = 1);
int compute_manifold_tree_height(float sigma_s, float sigma_r);
void build_manifolds_and_perform_filtering(const Image& f, const Image& f_joint, Image (*const phi)(const Image&, float, const vector<float>&),
    Image& sum_w_ki_Psi_blur, Image& sum_w_ki_Psi_blur_0, Image& min_pixel_dist_to_manifold_squared, Image& eta_k,
    Image& cluster_k, float sigma_s, float sigma_r, int current_tree_level, int tree_height, const vector<float>& phi_params, int num_pca_iter,
    vector<Image>* clusters_out, vector<Image>* manifolds_out);
Image h_filter(const Image& f, float sigma);
Image rf_filter(const Image& img, const Image& joint_img, float sigma_s, float sigma_r);
Image segment_image(const Image& img, const Image& cluster_k, int num_pca_iter);
Image scaleImage(const Image& img, float factor, Sampler option = Sampler::linear, const vector<float>& params = {});
Image resizeImage(const Image& img, int width, int height, Sampler option = Sampler::linear, const vector<float>& params = {});
Image copy(const Image img);
Image mulIm_cx1(const Image& multi_channel, const Image& single_channel);
Image divIm_cx1(const Image& multi_channel, const Image& single_channel);

Image phi_gaussian(const Image& dist_sq, float sigma, const vector<float>& unused);
Image non_local_basis(const Image& img, int radius);

#endif /* end of include guard: A10_H_PHUDVTKB */

