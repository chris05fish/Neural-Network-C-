#include<vector>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<cmath>
#include<cfloat>
#include<ctime>
using namespace std;

//#include "ann.h"

int main(int argc, char* argv[]) {
	srand( time(0) );

	string input = argv[1];//network structure
	string output = argv[2];
	vector<double> temp_w(100);
	for(int i = 0; i < 100; i++){
		int ind = rand() % 20 + 10;
		double aweight = 1.0 / (ind + 0.0);
		int sign = rand() % 2 + 1;
		if(sign == 1)
			aweight = -aweight;
		temp_w[i] = aweight; 
	}

	ifstream in;
	in.open(input.c_str());
	vector< int > points;
	int layer;
	in >> layer;
	while(!in.eof()){
		points.push_back(layer);
		in >> layer;	
	}//while
	in.close(); in.clear();
	ofstream outf;
	outf.open(output.c_str());

	for(int i = 0; i < points.size() - 1; i++){
		int total_w = points[i+1];//outgoing weights
		for(int j = 0; j < points[i] ; j++){
			for(int w = 0; w < total_w; w++){
				int ind = rand() % temp_w.size();
	outf << setiosflags(ios::showpoint) 
		<< setiosflags(ios::fixed) 
		<< setprecision(2) << temp_w[ind] << " ";
			}
			outf << endl;
		}//for j
			
	}
	outf.close(); outf.clear();	
}
