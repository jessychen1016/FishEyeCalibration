
#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "time.h"

const int BOARDWIDTH = 9;
const int BOARDHEIGHT = 6;

float SQUARESIZE = 17; 

using namespace std;
using namespace cv;

struct CalibSettings
{
    int getFlag()
    {
        int flag = 0;
        flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
        flag |= cv::fisheye::CALIB_FIX_SKEW;
        return flag;
    }

    Size getBoardSize()
    {
        return Size(BOARDWIDTH, BOARDHEIGHT);
    }

    float getSquareSize()
    {
        return SQUARESIZE;
    }
};

CalibSettings s;
static void
calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
}
vector<string>
getImageList(string path)
{
    vector<string> imagesName;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string tmpFileName = ent->d_name;
            if (tmpFileName.length() > 4) {
                auto nPos = tmpFileName.find(".png");
                if (nPos != string::npos) {
                    imagesName.push_back(path + '/' + tmpFileName);
                } else {
                    nPos = tmpFileName.find(".jpg");
                    if (nPos != string::npos)
                        imagesName.push_back(path + '/' + tmpFileName);
                }
            }
        }
        closedir(dir);
    }
    return imagesName;
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <pic path> [square size(mm)]" << endl;
        return 0;
    }
    if (argc == 3) {
        std::string size;
        size = argv[2];
        SQUARESIZE = std::stof(size);
    }

    string pathDirectory = argv[1];
    auto imagesName = getImageList(pathDirectory);
    
    vector<vector<Point2f>> imagePoints;
    Size imageSize;
    vector<vector<Point3f>> objectPoints;
    for (auto image_name : imagesName) {
        Mat view;
        view = imread(image_name.c_str());

        imageSize = view.size();
        vector<Point2f> pointBuf;
        // find the corners
        bool found = findChessboardCorners(view, s.getBoardSize(), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            imagePoints.push_back(pointBuf);
            drawChessboardCorners(view, s.getBoardSize(), Mat(pointBuf), found);
            cout << image_name << endl;
            namedWindow("image", CV_WINDOW_NORMAL);
            imshow("image", view);
            vector<Point3f> temp;
            calcBoardCornerPositions(s.getBoardSize(), s.getSquareSize(), temp);
            objectPoints.push_back(temp);
            cvWaitKey(0);
        } else {
            cout << image_name << " found corner failed! & removed!" << endl;
        }
    }
    
    cv::Matx33d cameraMatrix;
    cv::Vec4d distCoeffs;
    std::vector<cv::Vec3d> rvec;
    std::vector<cv::Vec3d> tvec;
    double rms = fisheye::calibrate(objectPoints, imagePoints, imageSize,cameraMatrix, distCoeffs, rvec, tvec, s.getFlag(), cv::TermCriteria(3, 20, 1e-6));
    
    cout << "-------------cameraMatrix--------------" << endl;
    cout << cameraMatrix << endl;
    printf("fx:                    %.13lf\nfy:                    %.13lf\ncx:                    %.13lf\ncy:                    %.13lf\n",
           cameraMatrix(0, 0),
           cameraMatrix(1, 1),
           cameraMatrix(0, 2),
           cameraMatrix(1, 2));

    cout << "---------------distCoeffs--------------" << endl;
    cout << distCoeffs << endl;
    for (auto image_name : imagesName) {

        Mat view = imread(image_name.c_str());
        Mat temp = view.clone();
        Mat intrinsic_mat(cameraMatrix), new_intrinsic_mat;
        intrinsic_mat.copyTo(new_intrinsic_mat);
        new_intrinsic_mat.at<double>(0, 0) *= 0.5; 
        new_intrinsic_mat.at<double>(1, 1) *= 0.5; 
        new_intrinsic_mat.at<double>(0, 2) = 0.5 * temp.cols; 
        new_intrinsic_mat.at<double>(1, 2) = 0.5 * temp.rows;
        cout<<new_intrinsic_mat<<endl;
        cout<< view.type()<< endl;
        cv::Mat map1, map2;

        Size size =  temp.size();
        fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Matx33d::eye(), new_intrinsic_mat, size, CV_16SC2, map1, map2 );
        
        auto start_time = clock();
        cv::remap(temp, view, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
        // fisheye::undistortImage(temp, view, cameraMatrix, distCoeffs,new_intrinsic_mat);
        auto end_time = clock();
        cout << "time in While  " << 1000.000*(end_time - start_time) / CLOCKS_PER_SEC << endl<< endl;
        // namedWindow("undist", CV_WINDOW_NORMAL);
        // imshow("undist", view);
        // waitKey(0);
        // haha
        
    }
    
    return 0;
}
