//
//  data_handling.cpp
//  ML_Dtree
//
//  Created by Shriram on 11/10/14.
//  Copyright (c) 2014 Shriram. All rights reserved.
//

#include "data_handling.h"
#include<iostream>
#include<string>
#include<stdio.h>
#include<fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <map>
#include <utility>
//#include "string_ops.h"

using namespace::std;

void print_x(vector<tr_example> a, int lim)
{   int j=0;
    vector<tr_example>::iterator i;
    for(i=a.begin(),j=0; (i!=a.end())&&(j<lim);i++,j++)
    {
        for(int j=0;j<15;j++)
            cout<<(*i).feature[j]<<" ";
        
        cout<<endl; 
    }
}

pair<long,int> finddim(char* a)
{
    
    int nf=0;
    long n=0;
    string s1;
    ifstream fp(a);
    
    s1.clear();
    
    getline(fp,s1,'\n');
    for (int i=0; i<s1.length(); i++) {
        if(s1[i]==',')
            nf++;
    }
    nf++;
    while(!fp.eof())
    {
        if(getline(fp,s1,'\n'))
            n++;
    }
    fp.close();
    
    return make_pair(n,nf);
}

void parse(string s,vector<string>* temp)
{   int prev=0,current=0;
    for(int i=0;i<s.length()+1;i++)
    {
        if(s[i]==','||s[i]=='\0')
        {
            prev=current;
            current=i;
            if(prev!=0)
                (temp)->push_back(s.substr(prev+1,current-prev-1));
            else
                (temp)->push_back(s.substr(prev,current-prev));
            
        }
    }
    
}

void fmapworker(string s1,int i,vector<map<string, long> > *b)
{
    map<string,long>::iterator j;
    vector<map<string, long> > a=*b;
    
    long nexf=a[i].begin()->second;
    
    if(a[i].size()==0)
      nexf=0;
    
    
    if((a[i].find(s1)==a[i].end()))
    {
        
        for(j=a[i].begin(); j!=a[i].end();j++)
        {if(j->second>nexf)
            nexf=j->second;
        }
        nexf++;
        a[i].insert(make_pair(s1,nexf));
        (*b).swap(a);
        
    }
    
    
    
}


void feature_map(char *a,data_unique *D1)
{
    string s1;
    ifstream fp(a);
    vector<map<string,long> > b=D1->names;
    vector<string> temp;
    map<string,long> c;
    
    if (b.size()==0)
    {
        for (int i=0; i<D1->nf; i++) {
            b.push_back(c);
        }
    }
    
    while(!fp.eof())
    {
        temp.clear();
        if(getline(fp,s1,'\n'))
        {
            for(int i=0;i<D1->nf;i++)
            {   //cout<<(b[i]).size()<<" ";
                if(((b[i]).size()>45)&&(D1->cont[i]==0))
                {
                    D1->cont[i]=1;
                    b[i].clear();
                    //cout<<"Feature marked continuous "<<i<<endl;
                }
                
            }
            //cout<<endl;
            parse(s1, &temp);
            
            for(int i=0;i<D1->nf;i++)
            {
               if(D1->cont[i]==0)
               fmapworker(temp[i], i, &b);
            }
            
        }
    }
    
    D1->names=b;
}


data_unique get_data(char* str)
{
    ifstream fp(str);
    tr_example data_temp;
    string s1,s2("?");
    data_unique Dummy(make_pair(0,0));
    FILE *p=fopen(str, "r");
    if (p==NULL) {
        cout<<"No such file exists"<<endl;
        cout<<"Place your file at "<<endl;
        system("pwd");
        exit(0);
        return Dummy;
        
    }
    
    data_unique D_tr(finddim(str));
    vector<string> temp;
    feature_map(str, &D_tr);
   
    
   
    while(!fp.eof())
    {   temp.clear();
        data_temp.feature.clear();
        for (int i=0;i<D_tr.nf ; i++) {
            data_temp.feature.push_back(0);
        }
        
        if(getline(fp,s1,'\n'))
        {
     
            //cout<<endl;
            parse(s1, &temp);
            
            for(int i=0;i<D_tr.nf;i++)
            {
                if(D_tr.cont[i]==0)
                    data_temp.feature[i]=(D_tr.names[i].find(temp[i]))->second;
                else
                    data_temp.feature[i]=atoi(temp[i].c_str());
            }
            if (s1.find(s2) == std::string::npos)
            {  D_tr.D.push_back(data_temp); }
            else
                D_tr.n--;
        }
    }
    
  
    cout<<"I seem to have read "<<D_tr.nf<<" features and "<<D_tr.n<<" examples"<<endl;
    D_tr.levels();

    
    return D_tr;
}

data_unique get_data(char* str,data_unique D)
{
    ifstream fp(str);
    tr_example data_temp;
    string s1,s2("?");
    data_unique Dummy(make_pair(0,0));
    FILE *p=fopen(str, "r");
    if (p==NULL) {
        cout<<"No such file exists"<<endl;
        cout<<"Place your file at "<<endl;
        system("pwd");
        exit(0);
        
    }
    
    data_unique D_tr(finddim(str));
    D_tr.names=D.names;
    
    vector<string> temp;
    //feature_map(str, &D_tr);
    
    while(!fp.eof())
    {   temp.clear();
        data_temp.feature.clear();
        for (int i=0;i<D_tr.nf ; i++) {
            data_temp.feature.push_back(0);
        }
        
        if(getline(fp,s1,'\n'))
        {
            
            //cout<<endl;
            parse(s1, &temp);
            
            for(int i=0;i<D_tr.nf;i++)
            {
                if(D_tr.cont[i]==0)
                    data_temp.feature[i]=(D_tr.names[i].find(temp[i]))->second;
                else
                    data_temp.feature[i]=atoi(temp[i].c_str());
            }
            if (s1.find(s2) == std::string::npos)
            {  D_tr.D.push_back(data_temp);}
            else
            {D_tr.n--;}
        }
    }
    
    
    //cout<<"I seem to have read "<<D_tr.nf<<" features and "<<D_tr.n<<" examples"<<endl;
    //D_tr.levels();
    
    
    return D_tr;
}

