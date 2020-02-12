#include <iostream>

#include <ctime>
#include "a10.h"

using namespace std;

// Prints out a tree of images (cluster or manifold tree) using the recursive labeling
// shown in the paper where left child has - appended and right child has + appended.
void write_image_tree(vector<Image>& images, string name, int start, int end, bool first = true)
{
    if (start >= end)
        return;

    string filename = "./Output/"+name+".png";
    if (first)
        filename = "./Output/"+name+"_0.png";
    images[start].write(filename);

    write_image_tree(images, name+"-", start+1, (end+start-1)/2+1, false);
    write_image_tree(images, name+"+", (end+start-1)/2+1, end, false);
}

// Test the output of the low_pass recursive filter for adaptive manifold construction.
// Passing this filter over an image constructs the first manifold for that image.
void h_filter_test()
{
    // Easiest way to test correctness of solution is to compare to output from the
    // original paper authors' MATLAB code, so we use one of their test images.
    Image im("./Input/kodim23.png");
    for (int sigma = 2; sigma < 64; sigma *= 2)
    {
        Image filtered = h_filter(im, sigma);
        filtered.write("./Output/h_filter_test_sigma"+to_string(sigma)+".png");
    }
}

// Test the output of the Domain Transform Recursive Filter for blurring over the adaptive manifolds
// by blurring an image over its first adaptive manifold.
void rf_filter_test()
{
    // Easiest way to test correctness of solution is to compare to output from the
    // original paper authors' MATLAB code, so we use one of their test images.
    Image im("./Input/kodim23.png");
    for (int r = 2; r < 64; r *= 2)
    for (int s = 2; s < 64; s *= 2)
    {
        Image filtered = rf_filter(im, h_filter(im, s), s, r/32.0f);
        filtered.write("./Output/rf_filter_test_s"+to_string(s)+"_r"+to_string(r)+".png");
    }
}

/* Construction of further adaptive manifolds depends on successful clustering of pixels using PCA
on their distance from previously generated manifolds. This test looks at the output of the binary
clustering of the input based on this criteria. The clustering step of the algorithm in general is
not independent from the manifold construction step, but looking at the first partition only gives
a pretty good idea how well the step is working on its own, and even for incorrect manifold construction
still produces reasonable looking results. Therefore we run the whole algorithm but only look at the
cluster output for this test. */
void clustering_test()
{
    vector<Image> clusters;
    Image im("./Input/kodim23.png");

    // We will generate clusters for bilateral filtering so we can easily compare to figures from
    // the authors' paper and to output from their sample MATLAB code if desired.
    float sigma_s = 24.0f;
    float sigma_r = 0.2f;
    adaptive_manifold_filter(im, sigma_s, sigma_r, phi_gaussian, im, false, &clusters);

    write_image_tree(clusters, "cluster", 0, clusters.size());
}

/* Constuction of new manifolds depends on successful clustering. It's hard to tell that the manifold
construction is working correctly without accumulating the final output, since an incorrect/poor choice
of manifold doesn't look much different than a good one when clustering is working correctly. We run the
whole algorithm, but only look at the manifolds produced at this step, since construction of the final
image from the manifolds is independent from their construction. */
void manifold_construction_test()
{
    vector<Image> manifolds;
    Image im("./Input/kodim23.png");

    // We will generate clusters for bilateral filtering so we can easily compare to figures from
    // the authors' paper and to output from their sample MATLAB code if desired.
    float sigma_s = 24.0f;
    float sigma_r = 0.2f;
    adaptive_manifold_filter(im, sigma_s, sigma_r, phi_gaussian, im, false, nullptr, &manifolds);

    write_image_tree(manifolds, "adaptive_manifold", 0, manifolds.size());
}

/* The simplest application of the adaptive manifold filter is to perform a gaussian filter using
the image color (range) values to jointly filter over the domain of the image. This implements
a classic bilateral filter. We test the speedup of the adaptive manifold method over the naive method.
Incorrect output but successful clustering and reasonable looking manifolds from the previous test
usually indicates a problem with the implementation of the h filter or the rf filter. */
void bilateral_filter_test(bool test_naive)
{
    // NOTE: Output looks noticably different than the naive bilateral filter implementation that was provided,
    // but looks almost identical to the output from the MATLAB code for the original paper.

    // Use a lower_resolution image when attempting to do the naive method, use image from author's sample
    // in order to compare to their work.
    string file_name = "./Input/kodim23.png";
    if (test_naive)
        file_name = "./Input/kodim23_low.png";
    Image im(file_name);
    float sigma_s = 24.0f;
    float sigma_r = 0.2f;

    cout << "-----Bilateral Filter Test Author Image-----" << endl;
    clock_t start = clock();
    if (test_naive)
        file_name = "./Output/adaptive_manifold_bilateral_low.png";
    else
        file_name = "./Output/adaptive_manifold_bilateral.png";
    adaptive_manifold_filter(im, sigma_s, sigma_r, phi_gaussian, im, false).write(file_name);
    clock_t end = clock();
    double am_duration = (end - start) * 1.0 / CLOCKS_PER_SEC;
    cout << "Adaptive Manifold Time: " << am_duration << " seconds" << endl;
    if (test_naive)
    {
        start = clock();
        bilateral(im, sigma_r, sigma_s).write("./Output/naive_bilateral_low.png");
        end = clock();
        double naive_duration = (end - start) * 1.0 / CLOCKS_PER_SEC;
        cout << "Naive Bilateral Time: " << naive_duration << " seconds" << endl;
        cout << "Adaptive Manifold Speed Up: " << naive_duration / am_duration << "X" << endl;
    }
    cout << "--------------------------------------------" << endl;

    // Test ability to denoise with bilateral filter on my own noisy, low-light image
    Image myIm("./Input/Noisy.png");
    sigma_s = 4.0f;
    sigma_r = 0.3f;
    cout << "-----Bilateral Denoise Test My Image-----" << endl;
    start = clock();
    adaptive_manifold_filter(myIm, sigma_s, sigma_r, phi_gaussian, myIm, false).write("./Output/adaptive_manifold_bila_denoise.png");
    end = clock();
    am_duration = (end - start) * 1.0 / CLOCKS_PER_SEC;
    cout << "Adaptive Manifold Time: " << am_duration << " seconds" << endl;
    if (test_naive)
    {
        start = clock();
        bilateral(myIm, sigma_r, sigma_s).write("./Output/naive_bilateral_denoise.png");
        end = clock();
        double naive_duration = (end - start) * 1.0 / CLOCKS_PER_SEC;
        cout << "Naive Bilateral Time: " << naive_duration << " seconds" << endl;
        cout << "Adaptive Manifold Speed Up: " << naive_duration / am_duration << "X" << endl;
    }
    cout << "-----------------------------------------" << endl;
}

/* Application to pseudo depth-of-field post-processing, showing potential usefulness for real-time
visual effects in computer graphics. */
void depth_of_field_test()
{
    // Note: depth map might not be linear, espcially if it was outputed by a traditional rasterization pipeline.
    // This may affect the appearance of the output, as regions toward the back will not be blurred the same as
    // regions toward the front even when visually they appear to be the same distance from the focal plane.

    Image subject("./Input/cube_full_focus.png");
    Image focal_dist("./Input/cube_focal_distance.png");
    Image depth("./Input/cube_depth_map.png");

    float sigma_s = 24.0f;
    float sigma_r = 0.2f;

    Image depth_based_blur = adaptive_manifold_filter(subject, sigma_s, sigma_r, phi_gaussian, depth, false);
    Image depth_of_field(subject.width(), subject.height(), subject.channels());
    for (int c = 0; c < subject.channels(); c++)
    for (int y = 0; y < focal_dist.height(); y++)
    for (int x = 0; x < focal_dist.width(); x++)
    {
        float alpha = focal_dist(x, y);
        depth_of_field(x, y, c) = (1.0f-alpha) * subject(x, y, c) + alpha * depth_based_blur(x, y, c);
    }

    depth_of_field.write("./Output/depth_of_field_example.png");

    // Test custom focal planes, given full focus image we can choose any distance for focal plane as post process.
    cout << "-----Pseudo Depth of Field Test-----" << endl;
    double avg = 0.0f;
    for (float fp = 0.0; fp <= 1.0; fp += 0.2)
    {
        clock_t start = clock();
        for (int c = 0; c < subject.channels(); c++)
        for (int y = 0; y < focal_dist.height(); y++)
        for (int x = 0; x < focal_dist.width(); x++)
        {
            float alpha = abs(fp - depth(x, y));
            depth_of_field(x, y, c) = (1.0f-alpha) * subject(x, y, c) + alpha * depth_based_blur(x, y, c);
        }
        depth_of_field.write("./Output/depth_of_field_focal_dist_"+to_string(fp)+".png");
        clock_t end = clock();
        avg += end - start;
    }
    double duration = avg / 5.0 / CLOCKS_PER_SEC;
    cout << "Depth of Field Time: " << duration << " seconds" << endl;
    cout << "-----------------------------------------" << endl;
}

/* Test of implementation of non-local means denoising by jointly filtering over the neighborhood of each pixel.
This shows that the algorithm works on very high-dimensional input, in this case a 149-dimensional filter. */
void non_local_means_test()
{
    Image im("./Input/Noisy.png"); // Testing on my own image.

    int patch_radius = 3;
    float sigma_s = 8.0f;
    float sigma_r = 0.35f;

    cout << "-----Non Local Means Denoise Test My Image-----" << endl;
    clock_t start = clock();
    Image nl_basis = non_local_basis(im, patch_radius); // nl_basis has 147 channels.

    int tree_height = 2 + compute_manifold_tree_height(sigma_s, sigma_r);

    adaptive_manifold_filter(im, sigma_s, sigma_r, phi_gaussian, nl_basis, true,
        nullptr, nullptr, tree_height, vector<float> {}, 2).write("./Output/non_local_means_denoise.png");
    clock_t end = clock();
    double duration = (end - start) * 1.0 / CLOCKS_PER_SEC;
    cout << "Non Local Means Time: " << duration << " seconds" << endl;
    cout << "-----------------------------------------" << endl;
}

int main()
{
    // Test your intermediate functions
    // srand(0);
    srand(time(0));

    //h_filter_test();
    //rf_filter_test();

    //clustering_test();
    //manifold_construction_test();

    //bilateral_filter_test(true);
    //bilateral_filter_test(false);
    depth_of_field_test();
    //non_local_means_test();

    return EXIT_SUCCESS;
}
