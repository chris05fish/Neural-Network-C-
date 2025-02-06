//Chris Fisher
#include<vector>
#include<iostream>
#include<iomanip>
#include<string>
#include<sstream>
#include<fstream>
#include<math.h>
#include<cfloat>
#include<pthread.h>
//#include<omp.h>
#include<time.h>
#include "ann.h"
using namespace std;
/*
typedef struct
{
        int i;
} threadParams_t;

pthread_t threads[100];
threadParams_t threadParams[100];

void *Ann::thread(void *threadp)
{
        threadParams_t *threadParams = (threadParams_t *)threadp;
        long double alpha = 0.01;
        for(int c=0; c<structure[threadParams->i+1]; c++)
        {
                for(int j=0; j<structure[threadParams->i]; j++)
                {
                        ann[threadParams->i][j].weights[c] = ann[threadParams->i][j].weights[c]+(alpha*ann[threadParams->i][j].a*ann[threadParams->i+1][c].delta);
                }
        }
        return((void *)0);
}
*/
Ann::Ann(vector<int> structure, string weights){
        ann.resize(structure.size());
        for(int i=0; i<(int)structure.size(); i++)
        {
                ann[i].resize(structure[i]);
        }
        long double weight;
        int c = 1;
        ifstream infile;
        infile.open (weights);
        while(c < (int)structure.size())
        {
            for(int i=0; i<structure[c-1]; i++)
            {
                    for(int j=0; j<structure[c]; j++)
                    {
                            infile >> weight;
                            ann[c-1][i].weights.push_back(weight);
                            if(c-1 != 0)
                                    ann[c-1][i].w = 0.01; //dont update if input node 
                    }
            }
            c++;
        }
        for(int i=0; i<structure[structure.size()-1]; i++)
                ann[structure.size()-1][i].w = 0.01;
        infile.close();
}
void Ann::backProp(vector<int> structure, string trainInput, string trainOutput, vector<vector<long double> > encode, int k)
{
        long double a0 = 1;
        int output;
        long double alpha = 0.01;

        //struct timespec start, end;
        //double fstart, fend;
        //double total = 0;

        //clock_gettime(CLOCK_MONOTONIC, &start);
        for(int it=0; it<k; it++)
        {
            ifstream infile;
            ifstream outfile;
            infile.open (trainInput);
            outfile.open (trainOutput);
            while(outfile >> output)
            {
                for(int i=0; i<structure[0]; i++)
                {
                    int a;
                    infile >> a;
                    ann[0][i].a = a;
                    //cout << "ann[0][" << i << "].a = " << a << endl;
                }
                //cout << "ex" << endl;
                //clock_gettime(CLOCK_MONOTONIC, &start);
                //#pragma omp parallel for num_threads(2)
                for(int i=0; i<(int)structure.size()-1; i++)
                {
                    for(int c=0; c<structure[i+1]; c++)
                    {
                        long double in = 0;
                        in = in + a0*ann[i+1][c].w;
                        //cout << "ann[i+1][c].w: (i+1)= " << i+1 << ", (c)= " << c << ", .w = " << ann[i+1][c].w << endl;
                        //#pragma omp parallel for num_threads(2)
                        for(int j=0; j<structure[i]; j++)
                        {
                            //cout << "in: " << in;
                            in = in + ann[i][j].a*ann[i][j].weights[c];
                            //cout << ", in: + " << ann[i][j].a << "*" << ann[i][j].weights[c] << endl; 
                        }
                        ann[i+1][c].in = in;
                        //cout << "ann[" << i+1 << "][" << c << "].in = " << showpoint << fixed << setprecision(12) << in << endl;
                        ann[i+1][c].a = 1/(1+exp(-in));
                        //cout << "ann[" << i+1 << "][" << c << "].a = " << showpoint << fixed << setprecision(12) << 1/(1+exp(-in)) << endl;
                    }
                }
                /*clock_gettime(CLOCK_MONOTONIC, &end);
                fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
                fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
                total = total + fend-fstart;*/
                //clock_gettime(CLOCK_MONOTONIC, &start);
                //#pragma omp parallel for num_threads(4)
                for(int i=0; i<structure[structure.size()-1]; i++)
                {
                    ann[structure.size()-1][i].delta = ann[structure.size()-1][i].a*(1-ann[structure.size()-1][i].a)*(encode[output][i]-ann[structure.size()-1][i].a);
                    //cout << "ann[" << structure.size()-1 << "][" << i << "].delta = " << showpoint << fixed << setprecision(12) << ann[structure.size()-1][i].a*(1-ann[structure.size()-1][i].a)*(encode[output][i]-ann[structure.size()-1][i].a) << endl;
                }
                /*clock_gettime(CLOCK_MONOTONIC, &end);
                fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
                fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
                total = total + fend-fstart;*/
                for(int i=structure.size()-2; i>=1; i--)
                {
                    for(int j=0; j<structure[i]; j++)
                    {
                            long double delta = 0;
                            for(int c=0; c<structure[i+1]; c++)
                            {
                                    delta = delta + ann[i+1][c].delta*ann[i][j].weights[c];
                            }
                            ann[i][j].delta = ann[i][j].a*(1-ann[i][j].a)*delta;
                            //cout << "ann[" << i << "][" << j << "].delta = " << showpoint << fixed << setprecision(12) << ann[i][j].a*(1-ann[i][j].a)*delta << endl;
                    }
                } 
                //clock_gettime(CLOCK_MONOTONIC, &start);
                //#pragma omp parallel for num_threads(4)
                for(int i=1; i<(int)structure.size(); i++)
                {
                    for(int j=0; j<structure[i]; j++)
                    {
                            ann[i][j].w = ann[i][j].w+(alpha*a0*ann[i][j].delta);
                            //cout << "ann[" << i << "][" << j << "].w = " << showpoint << fixed << setprecision(12) << ann[i][j].w+(alpha*a0*ann[i][j].delta) << endl;
                    }
                }
                /*clock_gettime(CLOCK_MONOTONIC, &end);
                fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
                fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
                total = total + fend-fstart;*/
                //clock_gettime(CLOCK_MONOTONIC, &start);
                //#pragma omp parallel for num_threads(4) default(none) reduction(+=ann) private(ann) shared(structure) collapse(3) schedule(dynamic)
                //#pragma omp parallel for num_threads(4) 
                for(int i=0; i<(int)structure.size(); i++)
                {
                        /*threadParams[i].i = i;
                        pthread_create(&threads[i], (void *)0, thread, (void *)&(threadParams[i]));
                        */
                        //int thread_count = omp_get_num_threads();
                        //cout << "Thread count: " << thread_count << endl;
                    for(int c=0; c<structure[i+1]; c++)
                    {
                            for(int j=0; j<structure[i]; j++)
                            {
                                    ann[i][j].weights[c] = ann[i][j].weights[c]+(alpha*ann[i][j].a*ann[i+1][c].delta);
                                    //int thread_count = omp_get_num_threads();
                                    //cout << "Thread count: " << thread_count << endl;
                                    //cout << "ann[" << i << "][" << j << "].weights[" << c << "] = " << showpoint << fixed << setprecision(12) << ann[i][j].weights[c]+(alpha*ann[i][j].a*ann[i+1][c].delta) << endl;
                            }
                    }
                }
                /*
                for(int i=0; i<(int)structure.size(); i++)
                {
                        pthread_join(threads[i], NULL);
                }*/
                /*clock_gettime(CLOCK_MONOTONIC, &end);
                fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
                fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
                total = total + fend-fstart;*/
                //cout << "Time = " << fend-fstart << " seconds." << endl;
            }
            infile.close();
            outfile.close();
        }
        /*clock_gettime(CLOCK_MONOTONIC, &end);
        fstart=start.tv_sec + (start.tv_nsec / 1000000000.0);
        fend=end.tv_sec + (end.tv_nsec / 1000000000.0);
        cout << "Time = " << total << " seconds." << endl;*/
        for(int i=0; i<structure[1]; i++)
                cout << showpoint << fixed << setprecision(12) << ann[0][0].weights[i] << " ";
        cout << endl;
}
void Ann::classification(vector<int> structure, string testInput, string testOutput, vector<vector<long double> > encode)
{//first 4 steps? If tie choose lower digit
        vector<int> dig;
        vector<int> out;
        long double right = 0;
        long double prec;
        long double a0 = 1;
        int output;
        ifstream infile;
        ifstream outfile;
        infile.open (testInput);
        outfile.open (testOutput);
        while(outfile >> output)
        {
            out.push_back(output);
            for(int i=0; i<structure[0]; i++)
            {
                int a;
                infile >> a;
                ann[0][i].a = a;
                //cout << "ann[0][" << i << "].a = " << a << endl;
            }
            //cout << "ex" << endl;
            for(int i=0; i<(int)structure.size()-1; i++)
            {
                for(int c=0; c<structure[i+1]; c++)
                {
                    long double in = 0;
                    in = in + a0*ann[i+1][c].w;
                    //cout << "ann[i+1][c].w: (i+1)= " << i+1 << ", (c)= " << c << ", .w = " << ann[i+1][c].w << endl;
                    for(int j=0; j<structure[i]; j++)
                    {
                        //cout << "in: " << in;
                        in = in + ann[i][j].a*ann[i][j].weights[c];
                        //cout << ", in: + " << ann[i][j].a << "*" << ann[i][j].weights[c] << endl;
                    }
                    ann[i+1][c].in = in;
                    //cout << "ann[" << i+1 << "][" << c << "].in = " << showpoint << fixed << setprecision(12) << in << endl;
                    ann[i+1][c].a = 1/(1+exp(-in));
                    //cout << "ann[" << i+1 << "][" << c << "].a = " << showpoint << fixed << setprecision(12) << 1/(1+exp(-in)) << endl;
                }
            }
            long double temp;
            long double min=DBL_MAX;
            int digit;
            for(int i=0; i<(int)encode.size(); i++)
            {
                temp = sqrt(pow((encode[i][0]-ann[structure.size()-1][0].a),2)+pow((encode[i][1]-ann[structure.size()-1][1].a),2)+pow((encode[i][2]-ann[structure.size()-1][2].a),2)+pow((encode[i][3]-ann[structure.size()-1][3].a),2)+pow((encode[i][4]-ann[structure.size()-1][4].a),2)+pow((encode[i][5]-ann[structure.size()-1][5].a),2)+pow((encode[i][6]-ann[structure.size()-1][6].a),2)+pow((encode[i][7]-ann[structure.size()-1][7].a),2)+pow((encode[i][8]-ann[structure.size()-1][8].a),2)+pow((encode[i][9]-ann[structure.size()-1][9].a),2));
                if(temp<min)
                {
                        min = temp;
                        digit = i;
                }
            }
            cout << digit << endl;
            dig.push_back(digit);
        }
        for(int i=0; i<(int)out.size(); i++)
        {
                if(out[i] == dig[i])
                        right = right + 1;
        }
        prec = right/out.size();
        cout << showpoint << fixed << setprecision(12) << prec << endl;
}
