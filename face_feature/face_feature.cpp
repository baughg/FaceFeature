// face_feature.cpp : Defines the entry point for the console application.
//
#include "face_detection.h"
#include "face_alignment.h"
#include "face_identification.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>


using namespace seeta;
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

int main(int argc, char** argv)
{
	// Initialize face detection model
	seeta::FaceDetection detector("model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("model/seeta_fa_v1.1.bin");
	const char* image_filename = "image/x5111zsv3vyi.bmp";

	//load image
	IplImage *img_grayscale = NULL;
	img_grayscale = cvLoadImage(image_filename, 0);
	if (img_grayscale == NULL)
	{
		return 0;
	}

	IplImage *img_color = cvLoadImage(image_filename, 1);
	int pts_num = 5;
	int im_width = img_grayscale->width;
	int im_height = img_grayscale->height;
	unsigned char* data = new unsigned char[im_width * im_height];
	unsigned char* data_ptr = data;
	unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
	int h = 0;
	for (h = 0; h < im_height; h++) {
		memcpy(data_ptr, image_data_ptr, im_width);
		data_ptr += im_width;
		image_data_ptr += img_grayscale->widthStep;
	}

	seeta::ImageData image_data;
	image_data.data = data;
	image_data.width = im_width;
	image_data.height = im_height;
	image_data.num_channels = 1;

	// Detect faces
	std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
	int32_t face_num = static_cast<int32_t>(faces.size());
	printf("faces: %u\n", face_num);
	if (face_num == 0)
	{
		delete[]data;
		cvReleaseImage(&img_grayscale);
		cvReleaseImage(&img_color);
		return 0;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark points[5];
	point_detector.PointDetectLandmarks(image_data, faces[0], points);

	// Visualize the results
	cvRectangle(img_color, cvPoint(faces[0].bbox.x, faces[0].bbox.y), cvPoint(faces[0].bbox.x + faces[0].bbox.width - 1, faces[0].bbox.y + faces[0].bbox.height - 1), CV_RGB(255, 0, 0));
	for (int i = 0; i<pts_num; i++)
	{
		cvCircle(img_color, cvPoint(points[i].x, points[i].y), 2, CV_RGB(0, 255, 0), CV_FILLED);
	}

	std::string out_filename = image_filename;
	out_filename.append(".jpg");
	cvSaveImage(out_filename.c_str(), img_color);
	// Release memory
	cvReleaseImage(&img_color);
	cvReleaseImage(&img_grayscale);
	delete[]data;

	// Id
	FaceIdentification face_recognizer("model/seeta_fr_v1.0.bin");
	// Create a image to store crop face.
	cv::Mat dst_img(face_recognizer.crop_height(),
		face_recognizer.crop_width(),
		CV_8UC(face_recognizer.crop_channels()));

	ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
	dst_img_data.data = dst_img.data;
	/* Crop Face */

	// read image
	cv::Mat src_img = cv::imread(image_filename, 1);
	EXPECT_NE(src_img.data, nullptr) << "Load image error!";

	// ImageData store data of an image without memory alignment.
	ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
	src_img_data.data = src_img.data;

	face_recognizer.CropFace(src_img_data, points, dst_img_data);
	cv::imwrite("image/crop.jpg", dst_img);
	/* Extract feature */
	std::vector<float> feature_desc(2048);
	face_recognizer.ExtractFeature(dst_img_data,&feature_desc[0]);
	
	std::string feature_filename = "feature.desc";

	if (argc >= 2) {
		feature_filename = std::string(argv[1]);
	}

	FILE* feat_file = NULL;
	feat_file = fopen(feature_filename.c_str(),"wb");

	if (feat_file)
	{
		fwrite(&feature_desc[0], sizeof(float), feature_desc.size(), feat_file);
		fclose(feat_file);
	}
	return 0;
}

