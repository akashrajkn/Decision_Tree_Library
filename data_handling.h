//
//  data_handling.h
//  ML_Dtree
//
//  Created by Shriram on 11/10/14.
//  Copyright (c) 2014 Shriram. All rights reserved.
//

#ifndef __ML_Dtree__data_handling__
#define __ML_Dtree__data_handling__

#include <stdio.h>
#include <map>
#include <vector>
#include <iostream>
using namespace::std;

typedef struct{
    /*
     int ed_num, age;
     long long fnlwgt;
     string workclass, education, mar_status, occupation, relationship, race, sex;
     long long capital_gain, capital_loss, hrs_per_week;
     string country, cls;
     */
    vector<long> feature;
    int y;
} tr_example;

struct dt_node
{
    double p;
    long long feature;   //integer corresponding to feature
    long long value; // Parent feature's value
    map<long long, dt_node*> children;// map of feature values to child pointers
    int no_children;
    int feature_done[14];
    dt_node* parent;
    double entropy;
    long long error;
    //vector<dt_node*> child;
    vector<tr_example> data;
};

struct data_unique
{
    long n;
    int nf;
    vector<map<string,long> > names;
    vector<int> cont;
    vector<tr_example> D;
    
    void levels()
    {   
        map<string,long>::iterator i;
        int j=0;
        for(j=0;j<nf;j++)
        {   if(cont[j]==0)
        {   cout<<"Feature "<<j<<" has levels of size "<< names[j].size()<<endl;
            cout<<"Levels are:"<<endl;
            for(i=names[j].begin();i!=names[j].end();i++)
                cout<<i->first<<":"<<i->second<<" ";
            cout<<endl<<endl;
            
        }
        else
        {
            cout<<"Feature "<<j<<" is continuous"<<endl<<endl;
        }
        }
    }
    
    data_unique(pair<long,int> b)
    {
        map<string,long> c;
        n=b.first;
        nf=b.second;
        for(int i=0;i<nf;i++)
        {
            cont.push_back(0);
            names.push_back(c);
        }
    }
};

data_unique get_data(char* str);
data_unique get_data(char* str,data_unique D);
void print_x(vector<tr_example> a,int lim);

#endif /* defined(__ML_Dtree__data_handling__) */
