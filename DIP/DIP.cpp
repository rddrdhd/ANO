// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdlib>

const int MAX_BRIGHTNESS = 255;
cv::Mat zero_lesson(cv::Mat src_8uc3_img) {
	cv::Mat gray_8uc3_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	cv::cvtColor(src_8uc3_img, gray_8uc3_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	gray_8uc3_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

	int x = 10, y = 15; // pixel coordinates

	uchar p1 = gray_8uc3_img.at<uchar>(y, x); // read grayscale value of a pixel, image represented using 8 bits
	float p2 = gray_32fc1_img.at<float>(y, x); // read grayscale value of a pixel, image represented using 32 bits
	cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x); // read color value of a pixel, image represented using 8 bits per color channel

	// print values of pixels
	printf("p1 = %d\n", p1);
	printf("p2 = %f\n", p2);
	printf("p3[ 0 ] = %d, p3[ 1 ] = %d, p3[ 2 ] = %d\n", p3[0], p3[1], p3[2]);

	gray_8uc3_img.at<uchar>(y, x) = 0; // set pixel value to 0 (black)

	// draw a rectangle
	cv::rectangle(gray_8uc3_img, cv::Point(65, 84), cv::Point(75, 94),
		cv::Scalar(50), CV_FILLED);

	// declare variable to hold gradient image with dimensions: width= 256 pixels, height= 50 pixels.
	// Gray levels wil be represented using 8 bits (uchar)
	cv::Mat gradient_8uc1_img(50, 256, CV_8UC1);

	// For every pixel in image, assign a brightness value according to the x coordinate.
	// This wil create a horizontal gradient.
	for (int y = 0; y < gradient_8uc1_img.rows; y++) {
		for (int x = 0; x < gradient_8uc1_img.cols; x++) {
			gradient_8uc1_img.at<uchar>(y, x) = x;
		}
	}

	cv::imshow("Gradient", gradient_8uc1_img);
	return gray_32fc1_img;
}

// helper func for string containing 
std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

// helper func for debugging matrix
void echo_matrix_stuff(cv::Mat mat) {
	std::string ty = type2str(mat.type());
	printf("%s %dx%d \n", ty.c_str(), mat.cols, mat.rows);
}


//================= 1 ==================
// Object Features Detection - indexing
// http://mrl.cs.vsb.cz//people/gaura/ano/features_01.pdf

// input: 8UC3 mat with grayscale values, value for decision
// output: 8uc3 mat with values 0 or 255
cv::Mat tresholding_objects(cv::Mat color_src_8uc3_img, int treshold_value) {
	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::Mat output_img(color_src_8uc3_img.rows, color_src_8uc3_img.cols, CV_8UC1);

	cv::cvtColor(color_src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	int current_value; // value (how much white 0-255) of curent pixel

	for (int y = 0; y < gray_8uc1_img.rows; y++) {
		for (int x = 0; x < gray_8uc1_img.cols; x++) {
			current_value = gray_8uc1_img.at<uchar>(y, x);
			if (current_value <= treshold_value) {
				output_img.at<uchar>(y, x) = 0;
			} else {
				output_img.at<uchar>(y, x) = 255;
			}
		}
	}
	return output_img;
}

// Recursive flood fill, depending on neighbour index
// input: mat, coordinates of examined value, old color and new color
void flood_fill(cv::Mat& mat, int x, int y, int old_color, int new_color){//((, cv::Mat output_img) {
	
	if (x < 1 || x >= mat.cols-1 || y < 1 || y >= mat.rows-1)
		return;
	if (mat.at<uchar>(y, x) != old_color)
		return;
	if (mat.at<uchar>(y, x) == new_color)
		return;

	mat.at<uchar>(y, x) = new_color;

	flood_fill(mat, x + 1, y, old_color, new_color);
	flood_fill(mat, x, y + 1, old_color, new_color);
	flood_fill(mat, x - 1, y, old_color, new_color);
	flood_fill(mat, x, y - 1, old_color, new_color);
	flood_fill(mat, x + 1, y + 1, old_color, new_color);
	flood_fill(mat, x + 1, y - 1, old_color, new_color);
	flood_fill(mat, x - 1, y + 1, old_color, new_color);
	flood_fill(mat, x - 1, y - 1, old_color, new_color);
	
}

// counting objects in indexed image
// input: mat
// output: number of unique values in mat
int count_unique_values(cv::Mat mat) {
	int histogram[MAX_BRIGHTNESS];
	int count = 0;
	for (int i = 0; i < 255; i++) {
		histogram[i] = 0;
	}
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			histogram[(int)mat.at<uchar>(y, x)]++;
		}
	}

	for (int i = 0; i < 255; i++) {
		if(histogram[i] != 0)
		count++;
	}
	return count;
}

// set each object unique index (1-255 value, 0 for background)
// input: 8UC3 mat with only 0 and 255 values
// output: 8UC3 mat with indexed objects (each object has unique value >1)
cv::Mat indexing_objects(cv::Mat tresholded_src_8uc3_img) {
	
	// indexing objects with flood method
	int my_index = 1;
	for (int y = 0; y < tresholded_src_8uc3_img.rows; y++) {
		for (int x = 0; x < tresholded_src_8uc3_img.cols; x++) {
			flood_fill(tresholded_src_8uc3_img, x, y, MAX_BRIGHTNESS, my_index);

			if (tresholded_src_8uc3_img.at<uchar>(y, x) == my_index) {
				my_index++;
			}
		}
	}

	return tresholded_src_8uc3_img; // indexed, we edited src mat recursively
}

// set each object with unique index different color
// input: 8uc3 mat with indexed objects
// output: 32fc3 mat with colored objects
cv::Mat coloring_objects(cv::Mat indexed_src_8uc3_img) {
	cv::Mat output_8uc3_img(indexed_src_8uc3_img.rows, indexed_src_8uc3_img.cols, CV_32FC3);

	// count objects and prepare random color for each
	cv::Vec3f* my_colors;
	int objects_count = count_unique_values(indexed_src_8uc3_img);
	my_colors = new cv::Vec3f[objects_count + 1];

	my_colors[0] = cv::Vec3f(0, 0, 0);
	for (int i = 1; i < objects_count + 1; i++) {
		my_colors[i] = cv::Vec3f(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
	}

	// fill my colorfull image
	for (int y = 0; y < indexed_src_8uc3_img.rows; y++) {
		for (int x = 0; x < indexed_src_8uc3_img.cols; x++) {
			int object_index = indexed_src_8uc3_img.at<uchar>(y, x);
			output_8uc3_img.at<cv::Vec3f>(y, x) = my_colors[object_index];
		}
	}

	for (int i = 1; i < objects_count+1; i++) {
		//std::cout << "[" << i << "] - R:" << my_colors[i][0] << ", G:" << my_colors[i][1] << ", B:" << my_colors[i][2] << std::endl;
	}
	return output_8uc3_img;
}

void cv1_tresholding_and_indexing() {
	cv::Mat src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	cv::Mat tresholded_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)
	cv::Mat indexed_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)
	cv::Mat colorful_32fc3_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	int const treshold_value = 100; // 0-255 value
	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);
	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);
	colorful_32fc3_img = coloring_objects(indexed_8uc1_img);
	// diplay images
	cv::imshow("Original 8uc3", src_8uc3_img);
	cv::imshow("Indexed 8uc1", tresholded_8uc1_img);
	cv::imshow("Colored 32fc3", colorful_32fc3_img);

	cv::waitKey(0); // wait until keypressed
}

//================= 2 ==================
// Object Features Detection - computing Moments and Features
// http://mrl.cs.vsb.cz//people/gaura/ano/features_02.pdf

// input: image with indexed objects, objects count
// output: array of images with one object 
cv::Mat* get_masks(cv::Mat indexed_8uc1_img, int objects_count) {
	cv::Mat* objects_masks = new cv::Mat[objects_count];
	for (int i = 1; i < objects_count; i++) {
		cv::Mat object_mask = cv::Mat::zeros(indexed_8uc1_img.size(), indexed_8uc1_img.type());
		for (int y = 0; y < indexed_8uc1_img.rows; y++) {
			for (int x = 0; x < indexed_8uc1_img.cols; x++) {
				if (indexed_8uc1_img.at<uchar>(y, x) == i) {
					object_mask.at<uchar>(y, x) = 255;
				}
			}
		}
		objects_masks[i] =object_mask;
		//cv::imshow("Mask of object " + i, object_mask);
	}
	return objects_masks;
}

// input: mask with one object
// output: number of pixels in object
int get_area(cv::Mat mask) {
	int area = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				area++;
			}
		}
	}
	return area;
}

// input: mask with one object
// output: circumference of object
int get_circumference(cv::Mat mask) {
	int circumference = 0;
	cv::Mat cannyImage;
	cv::Canny(mask, cannyImage, 128, 255, 3);
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (cannyImage.at<uchar>(y, x) == 255) {
				circumference++;
			}
		}
	}
	return circumference;
}

void cv2_moments_and_features() {
	cv::Mat src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	cv::Mat tresholded_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)
	cv::Mat indexed_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)
	cv::Mat colorful_32fc3_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)
	cv::Mat cannyImage; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	int const treshold_value = 100; // 0-255 value

	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);

	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);

	int objects_count = count_unique_values(indexed_8uc1_img);

	cv::Mat* objects_masks = get_masks(indexed_8uc1_img, objects_count);
	for (int i = 1; i < objects_count; i++) {
		std::cout << "Object " << i << ":\t"
			<< get_area(objects_masks[i]) << ",\t" 
			<< get_circumference(objects_masks[i]) << std::endl;
	}

	// diplay images
	//cv::imshow("Original 8uc3", src_8uc3_img);
	//cv::imshow("Indexed 8uc1", indexed_8uc1_img);
	colorful_32fc3_img = coloring_objects(indexed_8uc1_img);

	cv::imshow("Indexed", colorful_32fc3_img);
	cv::waitKey(0); // wait until keypressed
}

//================ MAIN =================
int main() {
// 1 - 10.02.2020
	//cv0_tresholding_and_indexing();
// 2 - 17.02.2020
	cv2_moments_and_features();

	return 0;
}
