//Chris Fisher
#include<string>
#include<iostream>
using namespace std;

#ifndef ANN_H
#define ANN_H

struct node{
        long double a; 
        long double in; 
        long double delta;
        long double w;//from 0-node i, change after updating
        vector<long double> weights; //weight from node to first index i to j
        node(){
                a = 0;
                in = 0;
                delta = 0;
                w = 0;
        };
};

class Ann {
        public:
                //void *thread(void *threadp);
                Ann(vector<int> structure, string weights);
                void backProp(vector<int> structure, string trainInput, string trainOutput, vector<vector<long double> > encode, int k);
                void classification(vector<int> structure, string testInput, string testOutput, vector<vector<long double> > encode);
        private:
                vector<vector<node> > ann; //node[0][0] = first input node, node[1][0] = first hidden layer node, so on

};

#endif

