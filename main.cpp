#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


Mat region_of_interest(Mat img_edges)
{
	//Region - of - interest vertices, 관심 영역 범위 계산시 사용 
	//We want a trapezoid shape, with bottom edge at the bottom of the image
	float trap_bottom_width = 0.85;  // width of bottom edge of trapezoid, expressed as percentage of image width
	float trap_top_width = 0.07;     // ditto for top edge of trapezoid
	float trap_height = 0.4;         // height of the trapezoid expressed as percentage of image height
	// Set ROI points(Trapezoid)
	int width = img_edges.cols;
	int height = img_edges.rows;
	Point points[4];
	points[0] = Point((width * (1 - trap_bottom_width)) / 2, height);
	points[1] = Point((width * (1 - trap_top_width)) / 2, height - height * trap_height);
	points[2] = Point(width - (width * (1 - trap_top_width)) / 2, height - height * trap_height);
	points[3] = Point(width - (width * (1 - trap_bottom_width)) / 2, height);


	Mat img_mask = Mat::zeros(img_edges.rows, img_edges.cols, CV_8UC1);

	Scalar ignore_mask_color = Scalar(255, 255, 255); // 검은색 처리
	const Point* ppt[1] = { points }; // 꼭지점을 갖고 있는 배열
	int npt[] = { 4 }; // 꼭지점의 갯수를 갖고 있는 배열

	//filling pixels inside the polygon defined by "vertices" with the fill color
	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

	//returning the image only where mask pixels are nonzero
	Mat img_masked;
	bitwise_and(img_edges, img_mask, img_masked); // bit 연산을 통한 이미지 중첩

	return img_masked;
}

// Filter the image to include only yellow and white pixels
void filter_colors(Mat _img_bgr, Mat& img_filtered)
{
	//차선 색깔 범위 
	Scalar lower_white = Scalar(200, 200, 200); //흰색 차선 (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //노란색 차선 (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	UMat img_bgr;
	_img_bgr.copyTo(img_bgr);
	UMat img_hsv, img_combine;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;

	//Filter white pixels
	inRange(img_bgr, lower_white, upper_white, white_mask);
	bitwise_and(img_bgr, img_bgr, white_image, white_mask);

	//Filter yellow pixels( Hue 30 )
	cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV);

	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(img_bgr, img_bgr, yellow_image, yellow_mask);

	//Combine the two above images
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, img_combine);

	img_combine.copyTo(img_filtered);
}

vector<Point2f> slidingWindow(Mat image, Rect window)
{
	vector<Point2f> points;
	const Size imgSize = image.size();
	bool shouldBreak = false;

	while (true)
	{
		float currX = window.x + window.width * 0.5f;
		Mat ROI = image(window);
		vector<Point> locations;
		//int count = countNonZero(ROI);
		//if(count < 0)
		//{
		findNonZero(ROI, locations);
		//}

		float avgX = 0.0f;
		for (int i = 0; i < locations.size(); ++i)
		{
			float x = locations[i].x;
			avgX += window.x + x;
		}
		avgX = locations.empty() ? currX : avgX / locations.size();

		Point point(avgX, window.y + window.height * 0.5f);
		points.push_back(point);
		window.y -= window.height;
		if (window.y < 0)
		{
			window.y = 0.;
			shouldBreak = true;
		}

		window.x += (point.x - currX);
		if (window.x < 0) window.x = 0;
		if (window.x + window.width >= imgSize.width) window.x = imgSize.width - window.width - 1;

		if (shouldBreak)break;
	}

	return points;
}

int main(int argc, char** argv) {

	// Create window
	VideoCapture cap;
	cap.open("challenge_video.mp4");

	if (!cap.isOpened()) {
		cerr << "Could not caputured! " << endl;
		return -1;
	}


	while (1) {
		Mat src;
		cap >> src;

		Mat img_filtered;
		filter_colors(src, img_filtered);
		imshow("filter", img_filtered);

		Mat img_gry, img_cny, img_ROI;

		cvtColor(img_filtered, img_gry, COLOR_BGR2GRAY);
		GaussianBlur(img_gry, img_gry, Size(3, 3), 0, 0);

		// Canny edge method
		Canny(img_gry, img_cny, 150, 255);

		img_ROI = region_of_interest(img_cny);
		imshow("ROI", img_ROI);

		// HoughTransformP 선분검출
		vector<Vec4i> linesP;
		HoughLinesP(img_ROI, linesP, 1, (CV_PI / 180), 40, 40, 10);
		Mat img_Houghp;
		img_ROI.copyTo(img_Houghp);
		
		cvtColor(img_Houghp, img_Houghp, COLOR_GRAY2BGR);

		
		for (size_t i = 0; i < linesP.size(); i++) {
			Vec4i l = linesP[i];
			line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 2, 8);
		}
		imshow("img_lane", src);

		if (waitKey(1) == 27)
			break;
	}
	cvDestroyAllWindows();
}
