// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdlib>

class img_object {       // The class
public:             // Access specifier
	cv::Mat mask;
	int object_class;
	int id;
	int area;        // Attribute (int variable)
	int xt;
	int yt;
	int circumference;
	double f1;
	double f2;
	cv::Vec3f class_color;
};

const int MAX_BRIGHTNESS = 255;
const int SQUARE = 1;
const int STAR = 2;
const int RECTANGLE = 3;
const int F1 = 0;
const int F2 = 1;
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
int get_center_of_mass_x(cv::Mat mask, int area) {
	int Mx = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
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
			if (mask.at<uchar>(y, x) == 255) {
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
			if (cannyImage.at<uchar>(y, x) == 255) {
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
			if (mask.at<uchar>(y, x) == 255) {
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

		img_object object;
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

void cv3_classification_using_talons() {
	cv::Mat src_8uc3_img, tresholded_8uc1_img, indexed_8uc1_img, colorful_32fc3_img, class_colored_orig, class_colored_new;
	int const treshold_value = 100; // 0-255 value
	int objects_count;
	cv::Vec2f etalon_SQUARE = cv::Vec2f(0, 0);
	cv::Vec2f eth¨alon_STAR = cv::Vec2f(0, 0);
	cv::Vec2f etalon_RECT = cv::Vec2f(0, 0);
	int star_count = 0;
	int rect_count = 0;
	int square_count = 0;
	cv::Vec3f my_color_STAR = cv::Vec3f(0,1, 1); //cv::Vec3f(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
	cv::Vec3f my_color_RECT = cv::Vec3f(0,0,1);//cv::Vec3f(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
	cv::Vec3f my_color_SQUARE = cv::Vec3f(1,1,0);//(((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)), ((double)rand() / (RAND_MAX)));
	double distance_to_RECT, distance_to_STAR, distance_to_SQUARE;

	src_8uc3_img = cv::imread("images/train.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	tresholded_8uc1_img = tresholding_objects(src_8uc3_img, treshold_value);
	indexed_8uc1_img = indexing_objects(tresholded_8uc1_img);
	objects_count = count_unique_values(indexed_8uc1_img);
	cv::Mat* objects_masks = get_masks(indexed_8uc1_img, objects_count);
	
	img_object* my_objects = new img_object[objects_count];
	img_object background;
	background.class_color = cv::Vec3f(0, 0, 0);
	my_objects[0] = background;
	// compute ehtalon for each object 
	for (int i = 1; i < objects_count; i++) {
		img_object object;
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
		std::cout << "Object " << i << ":\n\t"
			<< "area: " <<object.area << ",\t"
			<< "f1: "<<object.f1<< ",\t"
			<< "class: " << object.object_class << ",\n\t"
			<< "circ: " << object.circumference << ",\t"
			<< "f2: "<<object.f2<< ",\n\t" 
			//<< "center of the coord \tx:" << object.xt << ",\ty:" << object.yt << ",\n\t"
			<< std::endl;
		
		if (object.object_class == STAR) {
			eth¨alon_STAR[0] += object.f1;
			eth¨alon_STAR[1] += object.f2;
			star_count++;
			object.class_color = my_color_STAR;
		}
		if (object.object_class == SQUARE) {
			etalon_SQUARE[0] += object.f1;
			etalon_SQUARE[1] += object.f2;
			square_count++;
			object.class_color = my_color_SQUARE;
		}
		if (object.object_class == RECTANGLE) {
			etalon_RECT[0] += object.f1;
			etalon_RECT[1] += object.f2;
			rect_count++;
			object.class_color = my_color_RECT;
		}
		my_objects[i] = object;
	}

	eth¨alon_STAR = eth¨alon_STAR / star_count;
	etalon_RECT = etalon_RECT / rect_count;
	etalon_SQUARE = etalon_SQUARE / square_count;

	std::cout << "STAR(" << STAR << ") avg f1: " << eth¨alon_STAR[0] << ", f2: " << eth¨alon_STAR[1] << std::endl;
	std::cout << "SQUARE(" << SQUARE << ") avg f1: " << etalon_SQUARE[0] << ", f2: " << etalon_SQUARE[1] << std::endl;
	std::cout << "RECT("<<RECTANGLE<<") avg f1: " << etalon_RECT[0] << ", f2: " << etalon_RECT[1] << std::endl;

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

	img_object* new_my_objects = new img_object[new_objects_count];
	new_my_objects[0] = background;
	for (int i = 1; i < new_objects_count; i++) {
		img_object object;
		object.mask = new_objects_masks[i];
		object.area = get_area(new_objects_masks[i]);
		object.circumference = get_circumference(new_objects_masks[i]);
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
		distance_to_STAR = sqrt(pow(object.f2 - eth¨alon_STAR[F2], 2) + pow(object.f1 - eth¨alon_STAR[F1], 2));

		 // define nearest etalon
		if (distance_to_RECT < distance_to_SQUARE && distance_to_RECT < distance_to_STAR) {
			object.class_color = my_color_RECT;
			object.object_class = RECTANGLE;
		}
		if (distance_to_STAR < distance_to_SQUARE && distance_to_STAR < distance_to_RECT) {
			object.class_color = my_color_STAR;
			object.object_class = STAR;
		}
		if (distance_to_SQUARE < distance_to_RECT && distance_to_SQUARE < distance_to_STAR) {
			object.class_color = my_color_SQUARE;
			object.object_class = SQUARE;
		}
		new_my_objects[i] = object;

		// echo stuff
		std::cout << "new Object " << i << ":\n\t"
			//<< "area: " << object.area << ",\t"
			<< "f1: " << object.f1 << ",\t"
			<< "class: " << object.object_class << ",\n\t"
			//<< "circ: " << object.circumference << ",\t"
			<< "f2: " << object.f2 << ",\n\t"
			//<< "center of the coord \tx:" << object.xt << ",\ty:" << object.yt << ",\n\t"
			<< std::endl;
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
	}


	
		// diplay images
	//cv::imshow("Original 8uc3", src_8uc3_img);
	cv::imshow("Original classified", class_colored_orig);
	cv::imshow("new classified", class_colored_new);
	cv::waitKey(0); // wait until keypressed
}

//================ MAIN =================
int main() {
// 1 - 10.02.2020
	//cv1_tresholding_and_indexing();
// 2 - 17.02.2020
	//cv2_moments_and_features();
// 3 - 23.02.2020
	cv3_classification_using_talons();

	return 0;
}
