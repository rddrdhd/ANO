// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "backprop.h"
#include <cstdlib>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class MyObject {       // The class
public:             // Access specifier
	cv::Mat mask;
	int object_class;
	int id;
	int area;        // Attribute (int variable)
	int xt;
	int yt;
	int circumference;
	int nearest_centroid;
	double centroid_distance;
	double f1;
	double f2;
	cv::Vec3f class_color;
};

const int treshold_value = 100; // 0-255 value
const int k = 3;// number of centroids for k-means
const int plot_scale = 400; // resolution of plot

const int MAX_BRIGHTNESS = 255;
const int MIN_BRIGHTNESS = 0;
const int SQUARE = 1;
const int STAR = 2;
const int RECTANGLE = 3;
const int F1 = 0;
const int F2 = 1;
std::string object_names[3] = { "Ctverec", "Hvezda", "Obdelnik" };

const cv::Vec3f COLORS[3] = {cv::Vec3f(0.5, 0.5, 1),cv::Vec3f(1, 0.5, 0.5),cv::Vec3f(0.5, 1, 0.5)};//(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));

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

	gray_8uc3_img.at<uchar>(y, x) = MIN_BRIGHTNESS; // set pixel value to 0 (black)

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
				output_img.at<uchar>(y, x) = MIN_BRIGHTNESS;
			} else {
				output_img.at<uchar>(y, x) = MAX_BRIGHTNESS;
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
	for (int i = 0; i < MAX_BRIGHTNESS; i++) {
		histogram[i] = 0;
	}
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			histogram[(int)mat.at<uchar>(y, x)]++;
		}
	}

	for (int i = 0; i < MAX_BRIGHTNESS; i++) {
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

	my_colors[0] = cv::Vec3f(0, 0, 0); // black object "Background"
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

	//for (int i = 1; i < objects_count+1; i++) {
		//std::cout << "[" << i << "] - R:" << my_colors[i][0] << ", G:" << my_colors[i][1] << ", B:" << my_colors[i][2] << std::endl;
	//}
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
					object_mask.at<uchar>(y, x) = MAX_BRIGHTNESS;
				}
			}
		}
		objects_masks[i] =object_mask;
		//cv::imshow("Mask of object " + i, object_mask);
	}
	return objects_masks;
}
int get_area(cv::Mat mask) {
	int area = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
				area++;
			}
		}
	}
	return area;
}
int get_center_of_mass_x(cv::Mat mask, int area) {
	int Mx = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
				Mx += x;
			}
		}
	}
	return Mx/area;

}
int get_center_of_mass_y(cv::Mat mask, int area) {
	int My = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
				My += y;
			}
		}
	}
	return My/area;

}
int get_circumference(cv::Mat mask) {
	int circumference = 0;
	cv::Mat cannyImage;
	cv::Canny(mask, cannyImage, 128, 255, 3);
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (cannyImage.at<uchar>(y, x) == MAX_BRIGHTNESS) {
				circumference++;
			}
		}
	}
	return circumference;
}
double get_moment(cv::Mat mask, int p, int q, int xt, int yt) {
	double moment = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) =MAX_BRIGHTNESS) {
				moment += (double)(pow(x - xt, p)* pow(y - yt, q));
			}
		}
	}
	return moment;
}

void cv2_moments_and_features() {
	cv::Mat src_8uc3_img, tresholded_8uc1_img, indexed_8uc1_img, colorful_32fc3_img;
	int const treshold_value = 100; // 0-255 value
	int objects_count;

	src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);

	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);

	objects_count = count_unique_values(indexed_8uc1_img);

	cv::Mat* objects_masks = get_masks(indexed_8uc1_img, objects_count);

	for (int i = 1; i < objects_count; i++) {

		MyObject object;
		object.id = i;
		object.mask = objects_masks[i];
		object.area = get_area(objects_masks[i]);
		object.circumference = get_circumference(objects_masks[i]);
		object.xt = get_center_of_mass_x(objects_masks[i], object.area);
		object.yt = get_center_of_mass_y(objects_masks[i], object.area);
		object.f1 = ((double)object.circumference * object.circumference) / (100 * object.area);

		double m20 = get_moment(objects_masks[i], 2, 0, object.xt, object.yt);
		double m02 = get_moment(objects_masks[i], 0, 2, object.xt, object.yt);
		double m11 = get_moment(objects_masks[i], 1, 1, object.xt, object.yt);
		double mimin = (m20 + m02)/2 - sqrt((4 * m11 * m11) + (pow(m20 - m02, 2)))/2;
		double mimax = (m20 + m02)/2 + sqrt((4 * m11 * m11) + (pow(m20 - m02, 2)))/2;
		object.f2 = mimin / mimax;
		std::cout << "Object " << i << ":\n\t"
			<< "area: " <<object.area << ",\t"
			//<< "center of the coord \tx:"<<object.xt<<",\ty:" <<object.yt<< ",\n\t"
			<< "f1: "<<object.f1<< ",\n\t"
			<< "circ: " << object.circumference << ",\t"
			<< "f2: "<<object.f2<< ",\n\t" 
			<< std::endl;
	}

	// diplay images
	cv::imshow("Original 8uc3", src_8uc3_img);
	//cv::imshow("Indexed", indexed_8uc1_img);
	cv::waitKey(0); // wait until keypressed
}

//================= 3 ==================
// Object Classification Using Etalons
// http://mrl.cs.vsb.cz//people/gaura/ano/classification_using_etalons.pdf

void cv3_classification_using_etalons() {
	cv::Mat src_8uc3_img, tresholded_8uc1_img, indexed_8uc1_img, colorful_32fc3_img, class_colored_orig, class_colored_new;
	int const treshold_value = 100; // 0-255 value
	int objects_count;
	cv::Vec2f etalon_SQUARE = cv::Vec2f(0, 0);
	cv::Vec2f etalon_STAR = cv::Vec2f(0, 0);
	cv::Vec2f etalon_RECT = cv::Vec2f(0, 0);
	int star_count = 0;
	int rect_count = 0;
	int square_count = 0;
	double distance_to_RECT, distance_to_STAR, distance_to_SQUARE;

	src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);
	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);
	objects_count = count_unique_values(indexed_8uc1_img);
	cv::Mat* objects_masks = get_masks(indexed_8uc1_img, objects_count);
	
	MyObject* my_objects = new MyObject[objects_count];
	MyObject background;
	background.class_color = cv::Vec3f(0, 0, 0);
	my_objects[0] = background;
	// compute ehtalon for each object 
	for (int i = 1; i < objects_count; i++) {
		MyObject object;
		object.id = i;

		if (i < 5)
			object.object_class = SQUARE;
		else if (i < 9)
			object.object_class = STAR;
		else
			object.object_class = RECTANGLE;

		object.mask = objects_masks[i];
		object.area = get_area(objects_masks[i]);
		object.circumference = get_circumference(objects_masks[i]);
		object.xt = get_center_of_mass_x(object.mask, object.area);
		object.yt = get_center_of_mass_y(object.mask, object.area);
		object.f1 = ((double)object.circumference * object.circumference) / (100 * object.area);

		double m20 = get_moment(object.mask, 2, 0, object.xt, object.yt);
		double m02 = get_moment(object.mask, 0, 2, object.xt, object.yt);
		double m11 = get_moment(object.mask, 1, 1, object.xt, object.yt);
		double mimin = (m20 + m02)/2 - sqrt((4 * m11 * m11) + (pow(m20 - m02, 2)))/2;
		double mimax = (m20 + m02)/2 + sqrt((4 * m11 * m11) + (pow(m20 - m02, 2)))/2;
		object.f2 = mimin / mimax;

		// echo stuff
		std::cout << "Object " << i << ":\t" << "f1: "<<object.f1<< ",\t" << "f2: "<<object.f2<< ",\t" << "class: " << object.object_class  << std::endl;
		
		if (object.object_class == STAR) {
			etalon_STAR[F1] += object.f1;
			etalon_STAR[F2] += object.f2;
			star_count++;
			object.class_color = COLORS[STAR];
		}
		if (object.object_class == SQUARE) {
			etalon_SQUARE[F1] += object.f1;
			etalon_SQUARE[F2] += object.f2;
			square_count++;
			object.class_color = COLORS[SQUARE];
		}
		if (object.object_class == RECTANGLE) {
			etalon_RECT[F1] += object.f1;
			etalon_RECT[F2] += object.f2;
			rect_count++;
			object.class_color = COLORS[RECTANGLE];
		}
		my_objects[i] = object;
	}

	// compute etalon
	etalon_STAR = etalon_STAR / star_count;
	etalon_RECT = etalon_RECT / rect_count;
	etalon_SQUARE = etalon_SQUARE / square_count;

	// echo computed values
	std::cout << std::endl << "STAR(" << STAR << ") \t avg f1: " << etalon_STAR[F1] << ", \t f2: " << etalon_STAR[F2] << std::endl;
	std::cout << "SQUARE(" << SQUARE << ") \t avg f1: " << etalon_SQUARE[F1] << ", \t f2: " << etalon_SQUARE[F2] << std::endl;
	std::cout << "RECT("<<RECTANGLE<<") \t avg f1: " << etalon_RECT[F1] << ", \t f2: " << etalon_RECT[F2] << std::endl<< std::endl;

	// show colored img
	class_colored_orig = cv::Mat(indexed_8uc1_img.size(), CV_32FC3);
	// fill my original image with class colors
	for (int i = 1; i < objects_count; i++) {
		for (int y = 0; y < class_colored_orig.rows; y++) {
			for (int x = 0; x < class_colored_orig.cols; x++) {
				if (my_objects[i].mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
					class_colored_orig.at<cv::Vec3f>(y, x) = my_objects[i].class_color;
				}
			}
		}
	}

	// load new image
	cv::Mat analyze_8uc3_img = cv::imread("images/my_test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (analyze_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	cv::Mat new_tresholded_8uc1_img, new_indexed_8uc1_img;
	int new_objects_count;

	new_tresholded_8uc1_img = tresholding_objects(analyze_8uc3_img, treshold_value);
	
	new_indexed_8uc1_img = indexing_objects(new_tresholded_8uc1_img);
	new_objects_count = count_unique_values(new_indexed_8uc1_img);
	cv::Mat* new_objects_masks = get_masks(new_indexed_8uc1_img, new_objects_count);

	MyObject* new_my_objects = new MyObject[new_objects_count];
	new_my_objects[0] = background;
	for (int i = 1; i < new_objects_count; i++) {
		MyObject object;
		object.mask = new_objects_masks[i];
		object.area = get_area(object.mask);
		object.circumference = get_circumference(object.mask);
		object.xt = get_center_of_mass_x(object.mask, object.area);
		object.yt = get_center_of_mass_y(object.mask, object.area);
		object.f1 = ((double)object.circumference * object.circumference) / (100 * object.area);

		double m20 = get_moment(object.mask, 2, 0, object.xt, object.yt);
		double m02 = get_moment(object.mask, 0, 2, object.xt, object.yt);
		double m11 = get_moment(object.mask, 1, 1, object.xt, object.yt);
		double mimin = (m20 + m02) / 2 - sqrt((4 * m11 * m11) + (pow(m20 - m02, 2))) / 2;
		double mimax = (m20 + m02) / 2 + sqrt((4 * m11 * m11) + (pow(m20 - m02, 2))) / 2;
		object.f2 = mimin / mimax;

		// calculate distances from etalons - sqrt( y^2 + x^2 ) = distance
		distance_to_RECT = sqrt(pow(object.f2 - etalon_RECT[F2], 2) + pow(object.f1 - etalon_RECT[F1], 2));
		distance_to_SQUARE = sqrt(pow(object.f2 - etalon_SQUARE[F2], 2) + pow(object.f1 - etalon_SQUARE[F1], 2));
		distance_to_STAR = sqrt(pow(object.f2 - etalon_STAR[F2], 2) + pow(object.f1 - etalon_STAR[F1], 2));

		 // define nearest etalon
		if (distance_to_RECT < distance_to_SQUARE && distance_to_RECT < distance_to_STAR) {
			object.class_color = COLORS[RECTANGLE];
			object.object_class = RECTANGLE;
		}
		if (distance_to_STAR < distance_to_SQUARE && distance_to_STAR < distance_to_RECT) {
			object.class_color = COLORS[STAR];
			object.object_class = STAR;
		}
		if (distance_to_SQUARE < distance_to_RECT && distance_to_SQUARE < distance_to_STAR) {
			object.class_color = COLORS[SQUARE];
			object.object_class = SQUARE;
		}
		new_my_objects[i] = object;

		// echo stuff
		std::cout << "new Object " << i << ":\t" << "f1: " << object.f1 << ",\t" << "f2: " << object.f2 << "\t" << "class: " << object.object_class << std::endl;
	}

	// colorize new image by class
	class_colored_new = cv::Mat(analyze_8uc3_img.size(), CV_32FC3);
	for (int i = 1; i < new_objects_count; i++) {
		for (int y = 0; y < class_colored_new.rows; y++) {
			for (int x = 0; x < class_colored_new.cols; x++) {
				if (new_my_objects[i].mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
					class_colored_new.at<cv::Vec3f>(y, x) = new_my_objects[i].class_color;
				}
			}
		}
		cv::putText(class_colored_new, std::to_string(i), cvPoint(new_my_objects[i].xt, new_my_objects[i].yt),
			cv::FONT_HERSHEY_DUPLEX, 0.5, cvScalar(0), 2, CV_AA);
		cv::putText(class_colored_new, std::to_string(i), cvPoint(new_my_objects[i].xt, new_my_objects[i].yt),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255,255,255), 1, CV_AA);
	}
	for (int i = 1; i < objects_count; i++) {
		// put numbers on old img too
		cv::putText(class_colored_orig, std::to_string(i), cvPoint(my_objects[i].xt, my_objects[i].yt),
			cv::FONT_HERSHEY_DUPLEX, 0.5, cvScalar(0), 2, CV_AA);
		cv::putText(class_colored_orig, std::to_string(i), cvPoint(my_objects[i].xt, my_objects[i].yt),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 255, 255), 1, CV_AA);
	}
	
	const int plot_scale = 300;
	cv::Mat scatter_plot1 = cv::Mat(plot_scale, plot_scale, CV_32FC3);
	for (int i = 1; i < objects_count; i++) {
		cv::circle(scatter_plot1, cv::Point(my_objects[i].f1 * plot_scale, my_objects[i].f2 * plot_scale), 2, my_objects[i].class_color, 2);
	}
	for (int i = 1; i < new_objects_count; i++) {
		cv::circle(scatter_plot1, cv::Point(new_my_objects[i].f1 * plot_scale, new_my_objects[i].f2 * plot_scale), 3, new_my_objects[i].class_color, 1);
	}
	// diplay images
	cv::imshow("Original classified", class_colored_orig);
	cv::imshow("new classified", class_colored_new);
	//cv::imshow("Plot", scatter_plot1);
	cv::waitKey(0); // wait until keypressed
}

//================= 4 ==================
// k-means clusters
// http://mrl.cs.vsb.cz//people/gaura/ano/kmeans.pdf

MyObject* get_objects_with_moments(cv::Mat indexed_8uc1_img, int objects_count  ) {
	cv::Mat* objects_masks = get_masks(indexed_8uc1_img, objects_count);

	MyObject* my_objects = new MyObject[objects_count];
	MyObject background;
	background.class_color = cv::Vec3f(0, 0, 0);
	my_objects[0] = background;


	for (int i = 1; i < objects_count; i++) {

		double  m11, m02, m20, mimin, mimax;
		MyObject object;
		object.id = i;

		object.class_color = cv::Vec3f(1, 1, 1);

		object.mask = objects_masks[i];
		object.area = get_area(objects_masks[i]);
		object.circumference = get_circumference(objects_masks[i]);
		object.xt = get_center_of_mass_x(object.mask, object.area);
		object.yt = get_center_of_mass_y(object.mask, object.area);
		object.f1 = ((double)object.circumference * object.circumference) / (100 * object.area);

		m20 = get_moment(object.mask, 2, 0, object.xt, object.yt);
		m02 = get_moment(object.mask, 0, 2, object.xt, object.yt);
		m11 = get_moment(object.mask, 1, 1, object.xt, object.yt);
		mimin = (m20 + m02) / 2 - sqrt((4 * m11 * m11) + (pow(m20 - m02, 2))) / 2;
		mimax = (m20 + m02) / 2 + sqrt((4 * m11 * m11) + (pow(m20 - m02, 2))) / 2;
		object.f2 = mimin / mimax;

		//std::cout << "-- Object " << i << ":\t" << "f1: " << object.f1 << ",\t" << "f2: " << object.f2; if (i % 2 == 0) { std::cout << std::endl; }else { std::cout << "\t"; }

		my_objects[i] = object;
	}

	return my_objects;

}
void cv4_show_those_fckin_beautiful_imgs(cv::Mat indexed_8uc1_img, int objects_count, MyObject* my_objects, cv::Vec2f* centroids) {
	cv::Mat scatter_plot1 = cv::Mat(plot_scale, plot_scale, CV_32FC3);
	cv::Vec3f* class_colors = new cv::Vec3f[k];
	
	for (int i = 0; i < k; i++) {
		if (i < 3) {
			class_colors[i] = COLORS[i];
		}
		else {
			class_colors[i] = cv::Vec3f(0.5,0.5,0.5);//class_colors[i] = cv::Vec3f(((double)rand() / (RAND_MAX / 3) * 2), ((double)rand() / (RAND_MAX) / 3) * 2, ((double)rand() / (RAND_MAX / 3)));
		}
	}

	// show colored img
	cv::Mat class_colored = cv::Mat(indexed_8uc1_img.size(), CV_32FC3);
	// fill my original image with class colors
	for (int i = 1; i < objects_count; i++) {
		for (int y = 0; y < class_colored.rows; y++) {
			for (int x = 0; x < class_colored.cols; x++) {
				if (my_objects[i].mask.at<uchar>(y, x) == MAX_BRIGHTNESS) {
					class_colored.at<cv::Vec3f>(y, x) = class_colors[my_objects[i].object_class];
				}
			}
		}
	}

	// dotted chart
	for (int i = 1; i < objects_count; i++) {
		cv::circle(scatter_plot1, cv::Point(my_objects[i].f1 * plot_scale, my_objects[i].f2 * plot_scale), 2, class_colors[my_objects[i].object_class], 2);
	}

	for (int i = 1; i < objects_count; i++) {
		cv::putText(class_colored, std::to_string(i), cvPoint(my_objects[i].xt, my_objects[i].yt),
			cv::FONT_HERSHEY_DUPLEX, 0.5, cvScalar(0), 2, CV_AA);
		cv::putText(class_colored, std::to_string(i), cvPoint(my_objects[i].xt, my_objects[i].yt),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 255, 255), 1, CV_AA);

		cv::putText(class_colored, std::to_string(my_objects[i].object_class), cvPoint(my_objects[i].xt, my_objects[i].yt + 15),
			cv::FONT_HERSHEY_DUPLEX, 0.5, cvScalar(0), 2, CV_AA);
		cv::putText(class_colored, std::to_string(my_objects[i].object_class), cvPoint(my_objects[i].xt, my_objects[i].yt + 15),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(class_colors[my_objects[i].object_class][0], class_colors[my_objects[i].object_class][1], class_colors[my_objects[i].object_class][2]), 1, CV_AA);
	}

	for (int i = 0; i < k; i++) {
		cv::circle(scatter_plot1, cv::Point(centroids[i][F1] * plot_scale, centroids[i][F2] * plot_scale), 2, cv::Vec3f(1, 1, 1), 2);
	}

	for (int i = 0; i < k; i++) {
		cv::putText(scatter_plot1, std::to_string(i), cvPoint(centroids[i][F1] * plot_scale, centroids[i][F2] * plot_scale),
			cv::FONT_HERSHEY_DUPLEX, 0.5, cvScalar(0), 2, CV_AA);
		cv::putText(scatter_plot1, std::to_string(i), cvPoint(centroids[i][F1] * plot_scale, centroids[i][F2] * plot_scale),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 255, 255), 1, CV_AA);
	}

	// diplay images
	cv::imshow("Original", class_colored);
	cv::imshow("Plot", scatter_plot1);
}

void cv4_k_means() {
	cv::Mat src_8uc3_img, tresholded_8uc1_img, indexed_8uc1_img;
	int objects_count, nearest_centroid_index;

	int k_means_iteriation = 0;
	double distance, new_f1, new_f2;
	int* centroids_count = new int[k];

	src_8uc3_img = cv::imread("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);
	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);
	objects_count = count_unique_values(indexed_8uc1_img);
	
	MyObject* my_objects = get_objects_with_moments(indexed_8uc1_img, objects_count);

	// initialize 3 random centroids
	cv::Vec2f centroids[k];
	for (int j = 0; j < k; j++) {
		//int RandIndex = rand() % 4; //generates a random number between 0 and 3
  
		float x = ((double)rand() / (RAND_MAX));
		float y = ((double)rand() / (RAND_MAX));
		centroids[j] = cv::Vec2f(x, y);
	}

	bool done = false;
	do {
		std::cout<<std::endl<< " Iteration " << ++k_means_iteriation<<std::endl;

		// set nearest centroid for each object
		for (int i = 1; i < objects_count; i++) { //  for each object
			for (int j = 0; j < k; j++) { // for each centroid
				centroids_count[j] = 0; // initializing with zeros for later

				distance = (double)sqrt(pow(my_objects[i].f2 - centroids[j][F2], 2) + pow(my_objects[i].f1 - centroids[j][F1], 2));
				nearest_centroid_index = j;

				if (distance > my_objects[i].centroid_distance && my_objects[i].centroid_distance!=0) {
					distance = my_objects[i].centroid_distance;
					nearest_centroid_index = my_objects[i].object_class;
					done = false;
				}
				else {
					done = true;
				}

				my_objects[i].centroid_distance = distance;
				my_objects[i].object_class = nearest_centroid_index;
			}
			std::cout << std::fixed << std::setprecision(3)  << "\tObject " << i << ":\t" << "centroid = " << my_objects[i].object_class << ";\t" << "distance = " << my_objects[i].centroid_distance <<std::endl;
		}

		// move centroids and check if we are done
		std::cout << "\tMoving centroids: "<< std::endl;
		for (int j = 0; j < k; j++) { // for each centroid
			new_f1 = 0;
			new_f2 = 0;
			cv::Vec2f old_centroid = centroids[j];
			std::cout << std::setprecision(3) <<"\t Centroid "<< j <<" from " << centroids[j][F1] << ", " << centroids[j][F2];
			for (int i = 1; i < objects_count; i++) { //  for each object
				if (j == my_objects[i].object_class) { // if the centroid is object's nearest centroid
					new_f1 += my_objects[i].f1;
					new_f2 += my_objects[i].f2;
					centroids_count[j]++;
				}
			}
			if (centroids_count[j] == 0) centroids_count[j]++;
			new_f1 = new_f1 / centroids_count[j];
			new_f2 = new_f2 / centroids_count[j];
			centroids[j] = cv::Vec2f(new_f1, new_f2); // prumerne souradnice 
				
			std::cout << "\t"<< " to " << centroids[j][F1] << ", " << centroids[j][F2] << std::endl;

			// if centroids coordinates are the same for last 2 iterations, we are done
			done = (old_centroid == centroids[j]); 
		}
	} while (!done);

	std::cout << "\tThat's all Folks!";


	cv4_show_those_fckin_beautiful_imgs(indexed_8uc1_img, objects_count, my_objects, centroids);
	cv::waitKey(0); // wait until keypressed
}

//================= 4 ==================
// Neural network
// http://mrl.cs.vsb.cz//people/holusa/ano/cv6/bpnn.pdf

void train(NN* nn) {
	int n = 1000;
	double** trainingSet = new double* [n];
	double f1, f2, f3, a1, a2, a3, a4;
	std::ifstream neuralDataFile;
	neuralDataFile.open("dataclear.txt");
	if (!neuralDataFile) {
		std::cout << "Unable to open file";
		return;
	}
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
		neuralDataFile >> f1 >> f2 >> f3 >> a1 >> a2 >> a3 >> a4;
		int key = 0;
		for (auto value : { f1,f2,f3,a1,a2,a3,a4 }) {
			trainingSet[i][key] = value;
			++key;
		}
	}
	neuralDataFile.close();
	/*
	bool classA = i % 2;

	for (int j = 0; j < nn->n[0]; j++) {
		if (classA) {
			trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
		}
		else {
			trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
		}
	}

	trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
	trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;*/
	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i % n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}
void test(NN* nn, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	int num_err = 0;
	for (int n = 0; n < num_samples; n++)
	{
		bool classA = rand() % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (classA)
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}
		printf("predicted: %d\n", !classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
	}
	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}

void cv5_neural_network() {

	// prepare data to test & train
	const int TRESHOLD_VALUE = 100;
	cv::Mat src_8uc3_img, test_src_8uc3_img, train_img_tresholded, train_img_indexed, test_img_tresholded, test_img_indexed;
	src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	train_img_tresholded = tresholding_objects(src_8uc3_img, TRESHOLD_VALUE);
	train_img_indexed = indexing_objects(train_img_tresholded);

	test_src_8uc3_img = cv::imread("images/test01.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (test_src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	test_img_tresholded = tresholding_objects(test_src_8uc3_img, TRESHOLD_VALUE);
	test_img_indexed = indexing_objects(test_img_tresholded);

	int train_objects_count = count_unique_values(train_img_indexed);
	int test_objects_count = count_unique_values(test_img_indexed);
	MyObject* train_objects = new MyObject[train_objects_count];
	MyObject* test_objects = new MyObject[test_objects_count];

	//cv::imshow("Train img", train_img_indexed);
	//cv::imshow("Test img", test_img_indexed);

	train_objects = get_objects_with_moments(train_img_indexed, train_objects_count);
	test_objects = get_objects_with_moments(test_img_indexed, test_objects_count);

	NN* nn = createNN(3, 4, 4);
	train(nn);
	double* in = new double[nn->n[0]];
	// 
	for (int i = 0; i < test_objects_count; i++) {
		std::cout << "Object ["<<i<<"]:\tF1: "<< test_objects[i].f1 <<",\tF2:"<< test_objects[i].f2<<std::endl;
		test_img_indexed.at<uchar>(static_cast<int>(test_objects[i].yt), static_cast<int>(test_objects[i].xt)) = MAX_BRIGHTNESS;
	}




	cv::waitKey(0); // wait until keypressed


}
//================ MAIN =================
int main() {
// 1 - 10.02.2020
	//cv1_tresholding_and_indexing();
// 2 - 17.02.2020
	//cv2_moments_and_features();
// 3 - 23.02.2020
	//cv3_classification_using_etalons();
// 4 - 23.02.2020
	//cv4_k_means();
// 6
	cv5_neural_network();

	//cv::waitKey(0); // wait until keypressed
	return 0;
}
 