#include <iostream>

#include "a10.h"

using namespace std;

// Write your implementations here, or extend the Makefile if you add source
// files

// Applies the adaptive_manifold_filter to an image.
Image adaptive_manifold_filter(const Image& f, float sigma_s, float sigma_r, Image (*const phi)(const Image&, float, const vector<float>&),
    const Image& f_joint, bool adjust_outliers, vector<Image>* clusters_out, vector<Image>* manifolds_out, int tree_height,
    const vector<float>& phi_params, int num_pca_iter)
{
    // Image we are jointly filtering with must match dimensions.
    if (f.width() != f_joint.width() || f.height() != f_joint.height())
        throw InvalidArgument();

    int f_width = f.width();
    int f_height = f.height();
    int f_channels = f.channels();
    int joint_channels = f_joint.channels();

    if (tree_height <= 0)
        tree_height = compute_manifold_tree_height(sigma_s, sigma_r);
    int K = pow(2, tree_height) - 1;

    Image sum_w_ki_Psi_blur(f_width, f_height, f_channels);
    sum_w_ki_Psi_blur.set_color(0, 0, 0);

    Image sum_w_ki_Psi_blur_0(f_width, f_height, 1);
    sum_w_ki_Psi_blur_0.set_color(0);

    Image min_pixel_dist_to_manifold_squared(f_width, f_height);
    min_pixel_dist_to_manifold_squared.set_color(std::numeric_limits<float>::max());

    // Compute the first adaptive manifold
    Image eta_1 = h_filter(f_joint, sigma_s);
    Image cluster_1(f_width, f_height, 1);
    cluster_1.set_color(1);

    // For testing purposes, record clusters and manifolds if a return argument was provided.
    if (clusters_out)
        clusters_out->push_back(cluster_1);
    if (manifolds_out)
        manifolds_out->push_back(eta_1);

    int current_tree_level = 1;

    // Remaining manifolds computed recursively.
    build_manifolds_and_perform_filtering(f, f_joint, phi, sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0,
        min_pixel_dist_to_manifold_squared, eta_1, cluster_1, sigma_s, sigma_r, current_tree_level, tree_height,
        phi_params, num_pca_iter, clusters_out, manifolds_out);

    // Compute normalized filter result and adjust filter response for pixels
    // that are outliers from the adaptive manifolds.
    Image g = divIm_cx1(sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0);
    if (adjust_outliers)
        for (int c = 0; c < f_channels; c++)
        for (int y = 0; y < f_height; y++)
        for (int x = 0; x < f_width; x++)
            g(x, y, c) = f(x, y, c) +
                exp(-0.5f * min_pixel_dist_to_manifold_squared(x, y) / sigma_r / sigma_r) * (g(x, y, c) - f(x, y, c));

    return g;
}

// Computes the height of the manifold tree based on filter parameters.
int compute_manifold_tree_height(float sigma_s, float sigma_r)
{
    float Hs = floor(log2(sigma_s)) - 1;
    float Lr = 1 - sigma_r;

    return max(2.0f, ceil(Hs * Lr));
}

// Recursively constructs manifolds in the manifold tree and performs the filtering on each manifold.
void build_manifolds_and_perform_filtering(const Image& f, const Image& f_joint, Image (*const phi)(const Image&, float, const vector<float>&),
    Image& sum_w_ki_Psi_blur, Image& sum_w_ki_Psi_blur_0, Image& min_pixel_dist_to_manifold_squared, Image& eta_k,
    Image& cluster_k, float sigma_s, float sigma_r, int current_tree_level, int tree_height, const vector<float>& phi_params, int num_pca_iter,
    vector<Image>* clusters_out, vector<Image>* manifolds_out)
{
    // Image we are jointly filtering with must match dimensions.
    if (f.width() != f_joint.width() || f.height() != f_joint.height())
        throw InvalidArgument();

    int f_width = f.width();
    int f_height = f.height();
    int f_channels = f.channels();
    int joint_channels = f_joint.channels();

    // Compute downsampling factor for adaptive manifolds.
    float sigma_r_over_sqrt_2 = sigma_r / sqrt(2);
    float df = min(sigma_s / 4.0f, 256.0f * sigma_r);
    df = max(1.0f, pow(2.0f, floor(log2(df))));

    // Compute distance squared from manifold.
    Image X(f_width, f_height, joint_channels);
    Image pixel_dist_to_manifold_squared(f_width, f_height, 1);
    // Downsample manifold after computing distance if manifold not yet downsampled.
    if (eta_k.width() == f_width)
    {
        X = f_joint - eta_k;
        eta_k = scaleImage(eta_k, 1.0f/df);
    }
    // If manifold already downsampled, upsample temporarily for distance calculation.
    else
    {
        X = f_joint - resizeImage(eta_k, f_width, f_height);
    }

    for (int y = 0; y < f_height; y++)
    for (int x = 0; x < f_width; x++)
    {
        float dist_squared = 0.0f;
        for (int c = 0; c < joint_channels; c++)
            dist_squared += X(x, y, c) * X(x, y, c);
        pixel_dist_to_manifold_squared(x, y) = dist_squared;
    }

    // Perform splatting step, project pixel values onto the adaptive manifold.
    Image gaussian_distance_weights = phi(pixel_dist_to_manifold_squared, sigma_r_over_sqrt_2, phi_params);
    Image psi_splat = mulIm_cx1(f, gaussian_distance_weights);

    // Record minimum distance (squared) to manifold for each pixel to identify outliers later.
    for (int y = 0; y < f_height; y++)
    for (int x = 0; x < f_width; x++)
        min_pixel_dist_to_manifold_squared(x, y) =
            min(min_pixel_dist_to_manifold_squared(x, y), pixel_dist_to_manifold_squared(x, y));

    // Perform blurring step, perform filtering over the manifold.
    Image w_ki_psi_blur = rf_filter(scaleImage(psi_splat, 1.0f/df), eta_k, sigma_s / df, sigma_r_over_sqrt_2);
    Image w_ki_psi_blur_0 = rf_filter(scaleImage(gaussian_distance_weights, 1.0f/df), eta_k, sigma_s / df, sigma_r_over_sqrt_2);

    // Perform splicing step, gather blurred values from the manifold.
    sum_w_ki_Psi_blur = sum_w_ki_Psi_blur + mulIm_cx1(resizeImage(w_ki_psi_blur, f_width, f_height), gaussian_distance_weights);
    sum_w_ki_Psi_blur_0 = sum_w_ki_Psi_blur_0 + mulIm_cx1(resizeImage(w_ki_psi_blur_0, f_width, f_height), gaussian_distance_weights);

    // If we haven't reached the end of the tree, compute two new adaptive manifolds from the current manifold and recurse.
    if (current_tree_level < tree_height)
    {
        // Compute clusters for segmentation. The two segments will be used in computing the two new adaptive manifolds.
        Image dot = segment_image(X, cluster_k, num_pca_iter);

        Image cluster_minus(f_width, f_height, 1);
        Image cluster_plus(f_width, f_height, 1);
        for (int y = 0; y < f_height; y++)
        for (int x = 0; x < f_width; x++)
        {
            cluster_minus(x, y) = dot(x, y) < 0.0f && cluster_k(x, y);
            cluster_plus(x, y) = dot(x, y) >= 0.0f && cluster_k(x, y);
        }

        Image theta = (1.0f - gaussian_distance_weights);

        // Compute new adaptive manifold eta_minus
        Image theta_minus = cluster_minus * theta;
        Image eta_minus = divIm_cx1(
            h_filter(scaleImage(mulIm_cx1(f_joint, theta_minus), 1.0f/df), sigma_s / df),
            h_filter(scaleImage(theta_minus, 1.0f/df), sigma_s / df));

        // Compute new adaptive manifold eta_plus
        Image theta_plus = cluster_plus * theta;
        Image eta_plus = divIm_cx1(
            h_filter(scaleImage(mulIm_cx1(f_joint, theta_plus), 1.0f/df), sigma_s / df),
            h_filter(scaleImage(theta_plus, 1.0f/df), sigma_s / df));

        // For testing purposes, record clusters and manifolds if a return argument was provided.
        if (clusters_out)
        {
            clusters_out->push_back(cluster_minus);
            clusters_out->push_back(cluster_plus);
        }
        if (manifolds_out)
        {
            manifolds_out->push_back(eta_minus);
            manifolds_out->push_back(eta_plus);
        }

        // Recurse on new manifold eta minus
        build_manifolds_and_perform_filtering(f, f_joint, phi, sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0,
            min_pixel_dist_to_manifold_squared, eta_minus, cluster_minus, sigma_s, sigma_r, current_tree_level+1,
            tree_height, phi_params, num_pca_iter, clusters_out, manifolds_out);
        // Recurse on new manifold eta plus
        build_manifolds_and_perform_filtering(f, f_joint, phi, sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0,
            min_pixel_dist_to_manifold_squared, eta_plus, cluster_plus, sigma_s, sigma_r, current_tree_level+1,
            tree_height, phi_params, num_pca_iter, clusters_out, manifolds_out);
    }
}

// Applies a low-pass filter to create an adaptive manifold.
Image h_filter(const Image& f, float sigma)
{
    float a = exp(-sqrt(2.0f) / sigma);
    Image hx = copy(f);

    // Force filter to be symmetric and separable by applying it forwards and backwards in each dimension.

    // Horizontal
    for (int c = 0; c < f.channels(); c++)
    for (int y = 0; y < f.height(); y++)
    {
        // Left to right
        for (int x = 1; x < f.width(); x++)
        {
            hx(x, y, c) = f(x, y, c) + a * (hx(x-1, y, c) - f(x, y, c));
        }
        // Right to left
        for (int x = f.width()-2; x > -1; x--)
        {
            hx(x, y, c) = f(x, y, c) + a * (hx(x+1, y, c) - f(x, y, c));
        }
    }

    // Vertical
    Image out = copy(hx);
    for (int c = 0; c < f.channels(); c++)
    for (int x = 0; x < f.width(); x++)
    {
        // Top to bottom
        for (int y = 1; y < f.height(); y++)
        {
            out(x, y, c) = hx(x, y, c) + a * (out(x, y-1, c) - hx(x, y, c));
        }
        // Bottom to top
        for (int y = f.height()-2; y > -1; y--)
        {
            out(x, y, c) = hx(x, y, c) + a * (out(x, y+1, c) - hx(x, y, c));
        }
    }

    return out;
}

// Domain transform recursive filter, used to blurr over adaptive manifolds.
Image rf_filter(const Image& img, const Image& joint_img, float sigma_s, float sigma_r)
{
    // Image we are jointly filtering with must match dimensions.
    if (img.width() != joint_img.width() || img.height() != joint_img.height())
        throw InvalidArgument();

    int img_width = img.width();
    int img_height = img.height();
    int img_channels = img.channels();
    int joint_channels = joint_img.channels();

    Image dHdx(img_width, img_height, 1);
    Image dVdy(img_width, img_height, 1);

    float sigma_H = sigma_s;
    float Hs_sq = (sigma_H/sigma_s)*(sigma_H/sigma_s);
    float Hr_sq = (sigma_H/sigma_r)*(sigma_H/sigma_r);

    // Force filter to be symmetric and separable by applying it forwards and backwards in each dimension.

    // Horizontal domain transform
    for (int y = 0; y < img_height; y++)
    {
        dHdx(0, y) = sqrt(Hs_sq);
        for (int x = 1; x < img_width; x++)
        {
            float dIdx = 0.0f;
            for (int c = 0; c < joint_channels; c++)
                dIdx += (joint_img(x, y, c) - joint_img(x-1, y, c)) *
                    (joint_img(x, y, c) - joint_img(x-1, y, c));
            dHdx(x, y) = sqrt(Hs_sq + Hr_sq * dIdx);
        }
    }

    // Vertical domain transform
    for (int x = 0; x < img_width; x++)
    {
        dVdy(x, 0) = sqrt(Hs_sq);
        for (int y = 1; y < img_height; y++)
        {
            float dIdy = 0.0f;
            for (int c = 0; c < joint_channels; c++)
                dIdy += (joint_img(x, y, c) - joint_img(x, y-1, c)) *
                    (joint_img(x, y, c) - joint_img(x, y-1, c));
            dVdy(x, y) = sqrt(Hs_sq + Hr_sq * dIdy);
        }
    }

    int N = 1; // Increase iterations to make filter behave more like a gaussian than exponential.
    Image F = copy(img);

    // Filter over image
    for (int i = 0; i < N; i++)
    {
        float sigma_H_i = sigma_H * sqrt(3.0f) * pow(2, N - (i + 1)) / sqrt(pow(4, N) - 1);
        float a = exp(-sqrt(2.0f) / sigma_H_i);

        // Horizontal blur
        Image Fx = copy(F);
        for (int c = 0; c < img_channels; c++)
        for (int y = 0; y < img_height; y++)
        {
            // Left to right
            for (int x = 1; x < img_width; x++)
            {
                Fx(x, y, c) = F(x, y, c) + pow(a, dHdx(x, y)) * (Fx(x-1, y, c) - F(x, y, c));
            }
            // Right to left
            for (int x = img_width-2; x > -1; x--)
            {
                Fx(x, y, c) = F(x, y, c) + pow(a, dHdx(x+1, y)) * (Fx(x+1, y, c) - F(x, y, c));
            }
        }

        // Vertical blur
        Image Fy = copy(Fx);
        for (int c = 0; c < img_channels; c++)
        for (int x = 0; x < img_width; x++)
        {
            // Top to bottom
            for (int y = 1; y < img_height; y++)
            {
                Fy(x, y, c) = Fx(x, y, c) + pow(a, dVdy(x, y)) * (Fy(x, y-1, c) - Fx(x, y, c));
            }
            // Bottom to top
            for (int y = img_height-2; y > -1; y--)
            {
                Fy(x, y, c) = Fx(x, y, c) + pow(a, dVdy(x, y+1)) * (Fy(x, y+1, c) - Fx(x, y, c));
            }
        }

        F = Fy;
    }

    return F;
}

// Segments an image using PCA
Image segment_image(const Image& img, const Image& cluster_k, int num_pca_iter)
{
    // Reformat image into Nxc matrix, where N is the number of pixels
    // included in cluster_k and c is number of channels.
    int num_rows = 0;
    for (int y = 0; y < cluster_k.height(); y++)
    for (int x = 0; x < cluster_k.width(); x++)
        if (cluster_k(x, y))
            num_rows++;

    Eigen::MatrixXf X(num_rows, img.channels());
    for (int c = 0; c < img.channels(); c++)
    {
        int i = 0;
        for (int y = 0; y < img.height(); y++)
        for (int x = 0; x < img.width(); x++)
        {
            if (cluster_k(x, y))
            {
                X(i, c) = img(x, y, c);
                i++;
            }
        }
    }
    Eigen::MatrixXf X_t = X.transpose();

    // Multiple iterations converges random vector to the first eigenvector of
    // the covariance matrix X_t * X
    Eigen::MatrixXf v = Eigen::MatrixXf::Random(img.channels(), 1);
    for (int i = 0; i < num_pca_iter; i++)
        v = X_t * (X * v);

    v = v/v.norm(); // Normalize eigenvector

    // Sign of the dot product between pixels and the eigenvector
    // define the segmentation.
    Image segmentation(img.width(), img.height(), 1);
    for (int y = 0; y < segmentation.height(); y++)
    for (int x = 0; x < segmentation.width(); x++)
    {
        float dot = 0.0f;
        for (int c = 0; c < img.channels(); c++)
            dot += img(x, y, c) * v(c);
        segmentation(x, y) = dot;
    }

    return segmentation;
}

// Scale image by given factor
Image scaleImage(const Image& img, float factor, Sampler option, const vector<float>& params)
{
    switch (option)
    {
        case Sampler::nearest_neighbor:
            return scaleNN(img, factor);
        case Sampler::linear:
            return scaleLin(img, factor);
        case Sampler::bicubic:
            {
                float B = params.size() > 0 ? 1.0f/3.0f : params[0];
                float C = params.size() > 1 ? 1.0f/3.0f : params[1];
                return scaleBicubic(img, factor, B, C);
            }
        case Sampler::lanczos:
            {
                float a = params.size() ? 3.0f : params[0];
                return scaleLanczos(img, factor, a);
            }
    }

    return img;
}

// Scale image to specific size
Image resizeImage(const Image& img, int width, int height, Sampler option, const vector<float>& params)
{
    Image out(width, height, img.channels());
    float x_factor = ((float)width) / ((float)img.width());
    float y_factor = ((float)height) / ((float)img.height());

    // Have to reimplement scaling functions, modified TA solutions for each case.
    switch (option)
    {
        case Sampler::nearest_neighbor:
            {
                int ys, xs;
                for (int z=0; z<img.channels(); z++)
                for (int y=0; y<height; y++)
                for (int x=0; x<width; x++)
                {
                    ys = round(1/y_factor * y);
                    xs = round(1/x_factor * x);
                    out(x,y,z) = img.smartAccessor(xs,ys,z,true);
                }
            }
            break;
        case Sampler::linear:
            {
                float ys, xs;
                for (int z=0; z<img.channels(); z++)
                for (int y=0; y<height; y++)
                for (int x=0; x<width; x++)
                {
                    ys = 1/y_factor * y;
                    xs = 1/x_factor * x;
                    out(x,y,z) = interpolateLin(img, xs, ys, z);
                }
            }
            break;
        case Sampler::bicubic:
            {
                float B = params.size() > 0 ? 1.0f/3.0f : params[0];
                float C = params.size() > 1 ? 1.0f/3.0f : params[1];
                float A3 =   2 -      1.5f * B -  C;
                float A2 = - 3 +         2 * B +  C;
                float A0 =   1 -  0.33333f * B;
                float B3 = -0.166666f * B -     C;
                float B2 =              B + 5 * C;
                float B1 =         -2 * B - 8 * C;
                float B0 =  1.333333f * B + 4 * C;
                auto computeK = [&](float x)->float {
                    float kx = 0.0f;
                    float xabs = abs(x);
                    float x3 = pow(xabs, 3);
                    float x2 = pow(xabs, 2);
                    if( xabs < 1.0f) {
                      kx = A3 * x3 + A2 * x2 + A0;
                    }else if( 1 <= xabs && xabs < 2.0f)  {
                      kx = B3 * x3 + B2 * x2 + B1 * xabs + B0;
                    }
                    return kx;
                };
                for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                {
                    float ysrc = 1/y_factor * y;
                    float xsrc = 1/x_factor * x;
                    int xstart = (int) (floor(xsrc) - 2);
                    int xend   = (int) (floor(xsrc) + 2);
                    int ystart = (int) (floor(ysrc) - 2);
                    int yend   = (int) (floor(ysrc) + 2);
                    for(int xs = xstart; xs <= xend; ++xs)
                    for(int ys = ystart; ys <= yend; ++ys)
                    {
                        float w = computeK(xsrc - xs) * computeK(ysrc - ys);
                        for(int z=0; z<img.channels(); z++)
                            out(x,y,z) += img.smartAccessor(xs, ys, z, false) * w;
                    }
                }
            }
            break;
        case Sampler::lanczos:
            {
                float a = params.size() ? 3.0f : params[0];
                float PI2 = pow(M_PI,2);
                float PI_A = M_PI / a;
                auto computeK = [&](float x)->float {
                    float kx = 1.0f;
                    if( x!= 0.0f &&  -a <= x && x < a) {
                        kx = a * sin(M_PI * x) * sin(x * PI_A) /( PI2 * x * x);
                    }
                    return kx;
                };
                for(int y=0; y<height; y++)
                for(int x=0; x<width; x++)
                {
                    float ysrc = 1/y_factor * y;
                    float xsrc = 1/x_factor * x;
                    int xstart = (int) (floor(xsrc) - a + 1);
                    int xend   = (int) (floor(xsrc) + a);
                    int ystart = (int) (floor(ysrc) - a + 1);
                    int yend   = (int) (floor(ysrc) + a);
                    for(int xs = xstart; xs <= xend; ++xs)
                    for(int ys = ystart; ys <= yend; ++ys)
                    {
                        float w = computeK(xsrc - xs) * computeK(ysrc - ys);
                        for (int z=0; z<img.channels(); z++)
                          out(x,y,z) += img.smartAccessor(xs, ys, z, false) * w;
                    }
                }
            }
            break;
    }

    return out;
}

// Make a copy of an image
Image copy(const Image img)
{
    Image out(img.width(), img.height(), img.channels());
    for (int c = 0; c < img.channels(); c++)
    for (int y = 0; y < img.height(); y++)
    for (int x = 0; x < img.width(); x++)
        out(x, y, c) = img(x, y, c);
    return out;
}

// Multiply each channel of multi_channel by the first channel of single_channel
Image mulIm_cx1(const Image& multi_channel, const Image& single_channel)
{
    Image out(multi_channel.width(), multi_channel.height(), multi_channel.channels());
    for (int c = 0; c < out.channels(); c++)
    for (int y = 0; y < out.height(); y++)
    for (int x = 0; x < out.width(); x++)
        out(x, y, c) = single_channel(x, y) * multi_channel(x, y, c);
    return out;
}

// Divide each channel of multi_channel by the first channel of single_channel
Image divIm_cx1(const Image& multi_channel, const Image& single_channel)
{
    Image out(multi_channel.width(), multi_channel.height(), multi_channel.channels());
    for (int c = 0; c < out.channels(); c++)
    for (int y = 0; y < out.height(); y++)
    for (int x = 0; x < out.width(); x++)
        out(x, y, c) = multi_channel(x, y, c) / single_channel(x, y);
    return out;
}

// Implementation of phi for gaussian blur in range domain
Image phi_gaussian(const Image& dist_sq, float sigma, const vector<float>& unused)
{
    float sigma_sq = sigma * sigma;
    Image out(dist_sq.width(), dist_sq.height(), 1);
    for (int y = 0; y < out.height(); y++)
    for (int x = 0; x < out.width(); x++)
        out(x, y) = exp(-0.5f * dist_sq(x, y) / sigma_sq);
    return out;
}

// Computes the joint image data for performing non-local means filtering.
Image non_local_basis(const Image& img, int radius)
{
    int patch_size = 2*radius+1;
    int pixel_per_patch = patch_size * patch_size;
    Image mean_zero = img - img.mean();

    Image basis(img.width(), img.height(), pixel_per_patch*img.channels());
    for (int by = 0; by < basis.height(); by++)
    for (int bx = 0; bx < basis.width(); bx++)
    {
        int bc = 0;
        for (int pc = 0; pc < img.channels(); pc++)
        for (int py = -radius; py < radius+1; py++)
        for (int px = -radius; px < radius+1; px++)
        {
            float dist_sq = py*py + px*px;
            float weight = exp(-dist_sq / 2.0f / (radius / 2.0f));
            basis(bx, by, bc) = weight * mean_zero.smartAccessor(bx+px, by+py, pc, true);
            bc++;
        }
    }

    return basis;
}
