#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <sys/dir.h>
#include <cstdlib> 
#include <ctime>   
#include <algorithm> 

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "hog_visualization.cpp"
#include "opencv2/ml.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;

vector<Ptr<DTrees>> create_forest(int, int, int, int);
vector<Ptr<DTrees>> train_forest(vector<Ptr<DTrees>>, vector<Mat>, vector<int>, float [], int);
float predict_forest(vector<Ptr<DTrees>>, vector<Mat>, vector<int>, int, int, vector<int> *, vector<float> *);
vector<Mat> get_images(string, int, vector<int> *, bool, bool, bool, bool);
void arrayIndexes(vector<int>, vector<int>, vector<int>, vector<int>, vector<float>, vector<int> *, vector<int> *, vector<int> *, vector<float> *, int);
void non_max_suppression(vector<int>, vector<int>, vector<int> , vector<float>, vector<int> *);
float intersectionOverUnion(int, int, int, int, int, int, int);
void getRandomIndices(vector<int>, float [], vector<int> *);

//GLOBAL VARIABLES
int RESIZED_IMG = 160;
int BLOCK_SIZE = 40;
int BLOCK_STRIDE = 20;
int CELL_SIZE = 20;
int NUM_BINS = 9;


int main (int argc, char **argv){

	int i, j, k;

    //************************TASK 1***********************
    string task1_name = "./data/task1/obj1000.jpg";
	string path_name_task1 = "./data/task1/";

	Mat task1_img, task1_gray, img_tmp;

    task1_img = imread(task1_name, CV_LOAD_IMAGE_COLOR);   // Read the image
    if (!task1_img.data){
        printf("No image data.\n");
        return -1;
    }

    resize(task1_img, task1_img, Size(104, 104));
    imwrite(path_name_task1 + "./Test.jpg", task1_img);	//save img

    cvtColor(task1_img, task1_gray, COLOR_BGR2GRAY);		//make image gray
    imwrite(path_name_task1 + "./Gray_Image.jpg", task1_gray);	//save img

    namedWindow("Original window", WINDOW_AUTOSIZE);
 	namedWindow("Gray image", WINDOW_AUTOSIZE);

    imshow("Original window", task1_img);
    imshow("Gray image", task1_gray);


    //ROTATION IMAGE
    for(i=0;i<3;i++){
    	rotate(task1_img, img_tmp, i);
    	imwrite(path_name_task1 + "./Rotated_img_" + to_string(i) + ".jpg", img_tmp);
    }

    //FLIP IMAGE
    for(i=-1;i<2;i++){
    	flip(task1_img, img_tmp, i);
    	imwrite(path_name_task1 + "./Flipped_img_" + to_string(i+1) + ".jpg", img_tmp);
    }

    //CALCULATE BORDER
    int top, bottom, left, right;
    RNG rng(12345);
    top = (int) (0.1*task1_img.rows); bottom = (int) (0.1*task1_img.rows);
  	left = (int) (0.1*task1_img.cols); right = (int) (0.1*task1_img.cols);
    Scalar value( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
	copyMakeBorder(task1_img, img_tmp, top, bottom, left, right, BORDER_REPLICATE, value);
	imwrite(path_name_task1 + "./Bordered_1.jpg", img_tmp);
	
	//Calculate HOG DESCRIPTORS
	HOGDescriptor *hog = new HOGDescriptor(Size(task1_img.cols, task1_img.rows), Size(26,26), Size(13,13),	Size(13,13), 9);
	
	vector<float> features_hog;
    vector<Point> locations_hog;
	hog->compute(task1_img, features_hog, Size(32,32), Size(0,0), locations_hog);
	visualizeHOG(task1_img, features_hog, *hog, 3);
	waitKey(10);


	//************************TASK 2***********************
	//GETTING IMAGES AND LABELS
	vector<int> all_labels, test_labels;
	vector<Mat> all_imgs = get_images("./data/task2/train/0", 6, &all_labels, true, true, true, true);
	vector<Mat> test_imgs = get_images("./data/task2/test/0", 6, &test_labels, true, false, false, true);
	
	assert(all_imgs.size() == all_labels.size());
	assert(test_imgs.size() == test_labels.size());
	
	int num_trees = 1;
	int tree_depth = 21;
	float accuracy;
	int batch = all_imgs.size();

	vector<int> predicted_values;
	vector<float> perc_predicted;

	//************************************SINGLE TREE***********************************
	float batches_division_single[6] = {1, 1, 1, 1, 1, 1};  //PERCENTAGE OF IMGS TO TAKE FROM EACH CLASS

	vector<Ptr<DTrees>> single_tree = create_forest(num_trees, tree_depth, batch, 100);
    single_tree = train_forest(single_tree, all_imgs, all_labels, batches_division_single, num_trees);
    accuracy = predict_forest(single_tree, test_imgs, test_labels, num_trees, 6, &predicted_values, &perc_predicted);
    cout << accuracy << endl;

	//*************************FOREST***********************
	float batches_division_task2[6] = {0.87, 0.87, 0.87, 0.87, 0.87, 0.87};
	num_trees = 70;
	tree_depth = 25;
	batch = batches_division_task2[0] * all_imgs.size();

    vector<Ptr<DTrees>> forest_task2 = create_forest(num_trees, tree_depth, batch, 100);
    forest_task2 = train_forest(forest_task2, all_imgs, all_labels, batches_division_task2, num_trees);
    accuracy = predict_forest(forest_task2, test_imgs, test_labels, num_trees, 6, &predicted_values, &perc_predicted);
	cout << accuracy << endl;

	//**************************TASK 3*******************
  	vector<int> labels_det, test_labels_det;

	vector<Mat> imgs_det = get_images("./data/task3/train/0", 4, &labels_det, true, true, false, true);
	vector<Mat> test_imgs_det = get_images("./data/task3/test/", 1, &test_labels_det, false, false, false, false);


	//TRAIN FOREST
	float batches_division[4] = {0.73, 0.47, 0.5, 1};
	num_trees = 63;
	tree_depth = 100;
	batch = batches_division[0] * imgs_det.size();

    vector<Ptr<DTrees>> forest_det = create_forest(num_trees, tree_depth, batch, 100);
    forest_det = train_forest(forest_det, imgs_det, labels_det, batches_division, num_trees);

	int row, col;

	//SLIDING WINDOW
	int wind_size_n = 7;
	int windows_size[wind_size_n] = {85, 97, 108, 118, 126, 137, 150};
	int step_slide_n = 1;
    int step_slide[step_slide_n] = {20};
    Mat roi;

    //we don't have label of test data
    vector<int> useless_label;
    useless_label.push_back(0);

    vector<float> all_precisions, all_recalls;
    int num_gt = 3, tr;
    float thresholds[12] = {0.23, 0.31, 0.37, 0.44, 0.47, 0.51, 0.55, 0.61, 0.66, 0.73, 0.77, 0.83};

    //CLASS 0 BLUE, CLASS 1 GREEN, CLASS 2 RED
    int color_1[3] = {255, 255, 0};
    int color_2[3] = {0, 255, 0};
    int color_3[3] = {0, 255, 255};

    int total_box_predicted = 0;
	int correct_predicted = 0;
	float precision, recall;
	int num_imgs = test_imgs_det.size();

	//TRY DIFFERENT THRESHOLDS
    for(tr=0;tr<12;tr++){

    	float iou = 0;
    	total_box_predicted = 0;
		correct_predicted = 0;
		precision = 0;
		recall = 0;

    	//FOR ALL IMAGES, GET SLIDING BOX and FOR THIS CALCULATE PREDICTION
		for(i=0;i<num_imgs;i++){
			//I clone img, otherwise work for reference
	  		Mat DrawResultGrid = test_imgs_det[i].clone();

	  		vector<int> wind_cols, wind_rows, wind_sizes, predicted_val;
	  		vector<float> predicted_perc;

	  		//SLIDING WINDOWS
	  		for(j=0;j<wind_size_n;j++){
	  			for(k=0;k<step_slide_n;k++){
	  				for(row = 0; row <= test_imgs_det[i].rows - windows_size[j]; row += step_slide[k]){
			  			for(col = 0; col <= test_imgs_det[i].cols - windows_size[j]; col += step_slide[k]){
			  				
			  				//window in the img
				  			Rect windows(col, row, windows_size[j], windows_size[j]);
						   	roi = test_imgs_det[i](windows);

						   	vector<int> predicted_values_det;
							vector<float> perc_predicted_det;
							vector<Mat> roi_det;
							roi_det.push_back(roi);

			  				predict_forest(forest_det, roi_det, useless_label, num_trees, 4, &predicted_values_det, &perc_predicted_det);

			  				if(predicted_values_det[0] < 3 && perc_predicted_det[0]>thresholds[tr]){	//NOT BACKGROUND
			  					wind_cols.push_back(col);
			  					wind_rows.push_back(row);
			  					wind_sizes.push_back(windows_size[j]);
			  					predicted_val.push_back(predicted_values_det[0]);
			  					predicted_perc.push_back(perc_predicted_det[0]);
			  				}
						   	
			  			}
	  				}
	  			}
	  		}	

	  		//OPEN FILE WITH GROUND TRUTH
	  		stringstream gt_file;
	    	gt_file << std::setw(2) << std::setfill('0') << i;
	   		gt_file.str();
	  		string file_gt = "/home/francesco/TUM/Tracking Objects/Projects/2^/data/task3/gt/00" + gt_file.str() + ".gt.txt";

	  		ifstream infile(file_gt);
	  		string line;

	  		int top_x, top_y, bottom_x, bottom_y, tmp_val;
	  		vector<int> gt_top_x, gt_top_y, gt_bottom_x, gt_bottom_y;

		    if(infile){
		        while(getline(infile, line)){
		            sscanf(line.c_str(), "%d %d %d %d %d", &tmp_val, &top_x, &top_y, &bottom_x, &bottom_y);

		            gt_top_x.push_back(top_x);
		            gt_top_y.push_back(top_y);
		            gt_bottom_x.push_back(bottom_x);
		            gt_bottom_y.push_back(bottom_y);
		        }
		    }
		    else{
		    	cout << "Cannot open file." << endl;
		    	return -1;
		    }

	  		//NON MAX SUPPRESSION, I PERFORM IT ON EACH CLASS SEPARATELY
	    	for(j=0;j<3;j++){
	    		vector<int> picked;
	    		vector<int> chosen_wind_cols, chosen_wind_rows, chosen_wind_sizes;
	    		vector<float> chosen_scores;

	    		//TAKE ONLY WINDOWS of CLASS j
	    		arrayIndexes(predicted_val, wind_cols, wind_rows, wind_sizes, predicted_perc, 
	    			&chosen_wind_cols, &chosen_wind_rows, &chosen_wind_sizes, &chosen_scores, j);

	    		//NON MAX ONLY FOR WINDOWS of CLASS j
	    		non_max_suppression(chosen_wind_cols, chosen_wind_rows, chosen_wind_sizes, chosen_scores, &picked);

	    		total_box_predicted += picked.size();
			
				//AFTER NON MAX SUPPRESSION OF WINDOWS OF CLASS j
				for(k=0;k<picked.size();k++){
					Rect windows(chosen_wind_cols[picked[k]], chosen_wind_rows[picked[k]], 
						chosen_wind_sizes[picked[k]], chosen_wind_sizes[picked[k]]);

					//NUMBER TO 2 DIGITS
					stringstream stream;
					stream << fixed << setprecision(2) << chosen_scores[k];
					string percentage = stream.str();

					cv::putText(DrawResultGrid, 
				        "Class " + to_string(j) + ": " + percentage,
				        cv::Point(chosen_wind_cols[picked[k]], chosen_wind_rows[picked[k]]-5), //top-left position
				        cv::FONT_HERSHEY_DUPLEX,
				        0.4,
				        Scalar(color_1[j], color_2[j], color_3[j]), 
				        1);
					rectangle(DrawResultGrid, windows, Scalar(color_1[j], color_2[j], color_3[j]), 1, 8, 0);	

					iou = intersectionOverUnion(gt_top_x[j], gt_top_y[j], gt_bottom_x[j], gt_bottom_y[j], 
						chosen_wind_cols[picked[k]], chosen_wind_rows[picked[k]], chosen_wind_sizes[picked[k]]);

					if(iou > 0.5)
						correct_predicted++;
				}
	    	}

	     	imwrite("./result/" + to_string(i) + ".jpg", DrawResultGrid);
		}	

		precision = 0;
		if(total_box_predicted > 0)
			precision = float(correct_predicted) / float(total_box_predicted);


		recall = float(correct_predicted) / float(num_gt*num_imgs);
		if(recall>1)
			recall = 1;

		all_precisions.push_back(precision);
		all_recalls.push_back(recall);

	}

	ofstream myfile;
	myfile.open("pr_curve.txt");
	for(i=0;i<all_precisions.size(); i++){
		myfile << all_recalls[i] << ", " << all_precisions[i] << endl;
	}
	myfile.close();
	

	system("sort pr_curve.txt -k1 > pr_curve_points.txt");
	system("rm pr_curve.txt");


    return 0;
}


vector<Ptr<DTrees>> create_forest(int num_trees, int depth, int batch_size, int min_samp_count_perc){
	int i;
	vector<Ptr<DTrees>> trees;

	for(i=0; i<num_trees;i++){
		Ptr<DTrees> model = DTrees::create();
		model->setCVFolds(1);
		model->setMaxCategories(10);
		model->setMaxDepth(depth);
		model->setMinSampleCount(int(batch_size / min_samp_count_perc));

		trees.push_back(model);
	}

	return trees;
}

vector<Ptr<DTrees>> train_forest(vector<Ptr<DTrees>> forest, vector<Mat> imgs, vector<int> labels, float batches_division[], int num_trees){

	int rand_int, i, j;

	for(j=0;j<num_trees;j++){

		Mat feats, selected_labels, tmp;

		vector<float> features;
		vector<Point> locations;

		vector<int> selected_indices;
		/*I SELECT RANDOM INDICES PER EACH CLASS, ACCORDING TO PERCENTAGES FOR EACH CLASS
		BECAUSE THERE ARE CLASSES WITH MORE IMGS THAN OTHERS*/
		getRandomIndices(labels, batches_division, &selected_indices);

		for(i=0;i<selected_indices.size();i++){
			rand_int = selected_indices[i];	//SELECT RANDOM NUMBER
			Mat tmp = imgs[rand_int];	//select img from random index

		    HOGDescriptor *hog = new HOGDescriptor(
		    	Size(tmp.cols, tmp.rows),		//WIN SIZE
		    	Size(BLOCK_SIZE, BLOCK_SIZE),			//BLOCK SIZE
		    	Size(BLOCK_STRIDE,BLOCK_STRIDE),			//BLOCK STRIDE
		    	Size(CELL_SIZE,CELL_SIZE),			//CELL SIZE
		    	NUM_BINS						//# BINS
		    );
			hog->compute(tmp, features, Size(20,20), Size(0,0), locations);
			
			Mat tmp1;
			tmp1.push_back(features);
			transpose(tmp1, tmp1);

			feats.push_back(tmp1); 
			selected_labels.push_back(labels[rand_int]);
		}
		feats.convertTo(feats, CV_32F); 
		
		bool myTrainData = forest[j]->train(feats, ml::ROW_SAMPLE, selected_labels);
	}
	
	return forest;
}


float predict_forest(vector<Ptr<DTrees>> forest, vector<Mat> test_imgs, vector<int> labels, 
	int num_trees, int num_classes, vector<int> *predicted_values, vector<float> *perc_predicted){

	int i, j, correct = 0, wrong = 0, result_index;
	float perc;
	Mat tmp_img;

	assert(test_imgs.size() == labels.size());

	for(i=0;i<test_imgs.size();i++){
		vector<float> features;
		vector<Point> locations;

		tmp_img = test_imgs[i];
		resize(tmp_img, tmp_img, Size(160, 160));

		HOGDescriptor *hog = new HOGDescriptor(
	    	Size(tmp_img.cols, tmp_img.rows),
		    Size(BLOCK_SIZE, BLOCK_SIZE),	
		    Size(BLOCK_STRIDE,BLOCK_STRIDE),		
		    Size(CELL_SIZE,CELL_SIZE),		
		    NUM_BINS					
	    );
		hog->compute(tmp_img, features, Size(20,20), Size(0,0), locations);

		int scores[num_classes] = { 0 };
		for(j=0;j<num_trees;j++){
			int prediction;
			Mat tmp3;
			tmp3.push_back(features);
			tmp3.convertTo(tmp3, CV_32F); 

			prediction = forest[j]->predict(tmp3);
			scores[prediction]++;
		}
		result_index = distance(scores, max_element(scores, scores + sizeof(scores)/sizeof(scores[0])));
		predicted_values->push_back(result_index);

		perc = float(scores[result_index])/float(num_trees);
		perc_predicted->push_back(perc);

		if(labels[i] == result_index)
			correct++;
		else
			wrong++;
	}	
	//cout << "Correct: " << correct << " Misclassified: " << wrong << endl;
	
	return float(correct)/(wrong+correct);
}


vector<Mat> get_images(string dir_base, int n, vector<int> *labels, bool resized, bool augmentation, bool make_grey, bool folder_add){
	int i, j, k, x, y;
	DIR *dir;
	struct dirent *ent;
	Mat img_tmp, tmp2;

	vector<Mat> imgs;
	string dir_name;

	for(i=0;i<n;i++){
		if(folder_add == true)
			dir_name = dir_base + to_string(i);
		else
			dir_name = dir_base;

		if((dir = opendir(dir_name.c_str())) != NULL){

			vector <string> names;
			while ((ent = readdir(dir)) != NULL){
				string file_name = ent->d_name;
				if(file_name.compare(".") && file_name.compare(".."))
		    		names.push_back(file_name);
		    }
		    sort(names.begin(), names.end());

		    for(k=0;k<names.size();k++){
		    	img_tmp = imread(dir_name + "/" + names[k], CV_LOAD_IMAGE_COLOR);
		    	
		    	if (!img_tmp.data)
			        printf("No image data.\n");
			    else{
			    	//RESIZE IMAGE
			    	if(resized == true){
			    		resize(img_tmp, img_tmp, Size(RESIZED_IMG, RESIZED_IMG));
						imgs.push_back(img_tmp.clone());
			    		labels->push_back(i);				    		
			    	}
			    	else{	//NOT RESIZE
			    		imgs.push_back(img_tmp.clone());
			    		labels->push_back(i);
			    	}
			    	if(augmentation == true){
						for(x=-2;x<1;x+=2){
							Mat dst;
							if(x==-2)
								dst = img_tmp.clone();
							else{
								Mat src = img_tmp.clone();
								flip(src, dst, x);    
								imgs.push_back(dst.clone());
			    				labels->push_back(i);
							}

							for(y=0;y<3;y++){
								Mat src2 = dst.clone();
								Mat dst2; 
						    	rotate(src2, dst2, y);
						    	imgs.push_back(dst2.clone());
			    				labels->push_back(i);
			    			}
						}
			    	}
			    	if(make_grey == true){
			    		cvtColor(img_tmp, tmp2, COLOR_BGR2GRAY);
			    		imgs.push_back(tmp2.clone());
			    		labels->push_back(i);
			    	}
				}
		    }
			closedir(dir);
		}
		else
			perror("Cannot open directory.\n");
	}

	return imgs;
}


void arrayIndexes(vector<int> original_arr, vector<int> arr1, vector<int> arr2, vector<int> arr3, vector<float> arr4,
	vector<int> *res_1, vector<int> *res_2, vector<int> *res_3, vector<float> *res_4, int target){

	int i;

	assert(original_arr.size() == arr1.size());
	assert(original_arr.size() == arr2.size());
	assert(original_arr.size() == arr3.size());

	for(i=0;i<original_arr.size();i++){
		if(original_arr[i] == target){
			res_1->push_back(arr1[i]);
			res_2->push_back(arr2[i]);
			res_3->push_back(arr3[i]);
			res_4->push_back(arr4[i]);
		}
	}

}

void non_max_suppression(vector<int> cols, vector<int> rows, vector<int> wind_size, vector<float> scores, vector<int> *picked_indexes){

	assert(cols.size() == rows.size());
	assert(cols.size() == wind_size.size());
	assert(cols.size() == scores.size());

	if(cols.size() < 1)
		return;
	
	//x1 is cols, y1 is rows, x2 is cols + wind_size, y2 is rows + wind_size
	int i, j, last, pos;
	float overlap;

	//ARGSORT to take indices of ordinate score (Last index is highest score) => GREEDY APPROACH
	vector<int> indices(scores.size());
  	std::iota(indices.begin(), indices.end(), 0);
  	sort(indices.begin(), indices.end(),[&scores](int i1, int i2) {return scores[i1] < scores[i2];});


	while(indices.size() > 0){
		last = indices.size() - 1;
		i = indices[last];
		picked_indexes->push_back(i);

		vector<int> suppress;
		suppress.push_back(last);

		Rect2d wind1(cols[i], rows[i], wind_size[i], wind_size[i]);

		for(pos=0;pos<last;pos++){
			j = indices[pos];
			overlap = intersectionOverUnion(cols[i], rows[i], cols[i] + wind_size[i], rows[i] + wind_size[i], 
				cols[j], rows[j], wind_size[j]);

			Rect2d wind2(cols[j], rows[j], wind_size[j], wind_size[j]);
			Rect2d intersection = wind1 & wind2;

			if(overlap > 0.2)
				suppress.push_back(pos);
			else if(overlap > 0 && (intersection.area() == wind1.area() || intersection.area() == wind2.area()))
				suppress.push_back(pos);
		}
		
		//DELETE INDICES IN SUPPRESS
		sort(suppress.begin(), suppress.end());
		for(i=suppress.size()-1;i>=0;i--){
			indices.erase(indices.begin() + suppress[i]);
		}
	}

}


float intersectionOverUnion(int gt_top_x, int gt_top_y, int gt_bottom_x, int gt_bottom_y, 
	int wind_top_x, int wind_top_y, int wind_size){
	float iou, interArea;

	//SQUARE WINDOW
	assert((gt_bottom_y - gt_top_y) == (gt_bottom_x - gt_top_x));

	Rect2d wind(wind_top_x, wind_top_y, wind_size, wind_size);
	Rect2d gt_wind(gt_top_x, gt_top_y, gt_bottom_x - gt_top_x, gt_bottom_y - gt_top_y);

	Rect2d intersect = wind & gt_wind;

	//area of intersection rectangle
	interArea = intersect.area();

	//intersection area divided sum of prediction area, ground-truth, minus interesection area
	iou = interArea / float(wind.area() + gt_wind.area() - interArea);

	return iou;
}


void getRandomIndices(vector<int> labels, float percentages_batches[], vector<int> *all_indices){

	int i, j, old_lab = -1;
	int total_per_class, batch_imgs = 0;
	vector<int> labels_offset;

	for(i=0;i<labels.size();i++){
		if(old_lab != labels[i]){
			labels_offset.push_back(i);
			old_lab = labels[i];
		}
	}
	labels_offset.push_back(labels.size());

	for(i=0;i<labels_offset.size()-1;i++){
		vector<int> indices;

		for(j=labels_offset[i];j<labels_offset[i+1];j++)
			indices.push_back(j);


		srand(time(NULL));    
    	random_shuffle(indices.begin(), indices.end()); 

    	total_per_class = labels_offset[i+1] - labels_offset[i];
    	batch_imgs = percentages_batches[i] * total_per_class;

    	for(j=0;j<batch_imgs;j++)
    		all_indices->push_back(indices[j]);
    }

}