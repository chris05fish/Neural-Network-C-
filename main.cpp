//Chris Fisher
#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<fstream>
//#include<omp.h>
#include<time.h>
#include "ann.h"
using namespace std;
int main(int argc, char* argv[])
{
        string trainInput = argv[1];
        string trainOutput = argv[2];
        string testInput = argv[3];
        string testOutput = argv[4];
        string struc = argv[5];
        string weights = argv[6];
        int k = atoi(argv[7]);
        vector<int> structure;
        vector<vector<long double> > encode;
        //create object of ann(vectore<struct>) to see how many nodes 
        ifstream infile;
        int temp;
        string tmpstring;
        infile.open (struc); 
        while(getline(infile,tmpstring))
        {
                temp = stoi(tmpstring);
                structure.push_back(temp);
                //mystring+=tmpstring;
        }
        infile.close();
        encode.resize(10);
        for(int i=0; i<(int)encode.size(); i++)
                encode[i].resize(10);
        encode[0][0] = 0.1;
        encode[0][1] = 0.9;
        encode[0][2] = 0.9;
        encode[0][3] = 0.1;
        encode[0][4] = 0.9;
        encode[0][5] = 0.9;
        encode[0][6] = 0.1;
        encode[0][7] = 0.9;
        encode[0][8] = 0.9;
        encode[0][9] = 0.1;
        encode[1][0] = 0.9;
        encode[1][1] = 0.1;
        encode[1][2] = 0.9;
        encode[1][3] = 0.9;
        encode[1][4] = 0.1;
        encode[1][5] = 0.9;
        encode[1][6] = 0.9;
        encode[1][7] = 0.1;
        encode[1][8] = 0.9;
        encode[1][9] = 0.9;
        encode[2][0] = 0.9;
        encode[2][1] = 0.9;
        encode[2][2] = 0.1;
        encode[2][3] = 0.9;
        encode[2][4] = 0.9;
        encode[2][5] = 0.1;
        encode[2][6] = 0.9;
        encode[2][7] = 0.9;
        encode[2][8] = 0.1;
        encode[2][9] = 0.9;
        encode[3][0] = 0.1;
        encode[3][1] = 0.1;
        encode[3][2] = 0.9;
        encode[3][3] = 0.9;
        encode[3][4] = 0.1;
        encode[3][5] = 0.1;
        encode[3][6] = 0.9;
        encode[3][7] = 0.9;
        encode[3][8] = 0.1;
        encode[3][9] = 0.1;
        encode[4][0] = 0.9;
        encode[4][1] = 0.1;
        encode[4][2] = 0.1;
        encode[4][3] = 0.9;
        encode[4][4] = 0.9;
        encode[4][5] = 0.1;
        encode[4][6] = 0.1;
        encode[4][7] = 0.9;
        encode[4][8] = 0.9;
        encode[4][9] = 0.1;
        encode[5][0] = 0.9;
        encode[5][1] = 0.9;
        encode[5][2] = 0.1;
        encode[5][3] = 0.1;
        encode[5][4] = 0.9;
        encode[5][5] = 0.9;
        encode[5][6] = 0.1;
        encode[5][7] = 0.1;
        encode[5][8] = 0.9;
        encode[5][9] = 0.9;
        encode[6][0] = 0.1;
        encode[6][1] = 0.9;
        encode[6][2] = 0.1;
        encode[6][3] = 0.1;
        encode[6][4] = 0.9;
        encode[6][5] = 0.1;
        encode[6][6] = 0.1;
        encode[6][7] = 0.9;
        encode[6][8] = 0.1;
        encode[6][9] = 0.1;
        encode[7][0] = 0.9;
        encode[7][1] = 0.1;
        encode[7][2] = 0.1;
        encode[7][3] = 0.9;
        encode[7][4] = 0.1;
        encode[7][5] = 0.1;
        encode[7][6] = 0.9;
        encode[7][7] = 0.1;
        encode[7][8] = 0.1;
        encode[7][9] = 0.9;
        encode[8][0] = 0.1;
        encode[8][1] = 0.1;
        encode[8][2] = 0.9;
        encode[8][3] = 0.1;
        encode[8][4] = 0.1;
        encode[8][5] = 0.9;
        encode[8][6] = 0.1;
        encode[8][7] = 0.1;
        encode[8][8] = 0.9;
        encode[8][9] = 0.1;
        encode[9][0] = 0.1;
        encode[9][1] = 0.9;
        encode[9][2] = 0.9;
        encode[9][3] = 0.9;
        encode[9][4] = 0.1;
        encode[9][5] = 0.1;
        encode[9][6] = 0.1;
        encode[9][7] = 0.9;
        encode[9][8] = 0.9;
        encode[9][9] = 0.9;
        Ann net(structure, weights);
//#pragma omp parallel num_threads(2)
        net.backProp(structure, trainInput, trainOutput, encode, k);
        for(int i=0; i<10; i++)
            net.classification(structure, testInput, testOutput, encode);
        return 0;
}
