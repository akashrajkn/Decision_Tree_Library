#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <map>
#include <utility>
#include "data_handling.h"
//#include "misc_functions.h"
//#include "dec_tree.h"
using namespace std;

#define ALPHA 0.005
#define THRESHOLD 0.25
long long levels = 0;
dt_node dt_root;
long N;


void print_tree(dt_node node, long long levels)
{
    //dt_node temp;
    map<long long,dt_node*>::iterator it;
    //if(node.feature!=-1)
    if(node.parent!=NULL)
        cout<<node.feature<<" "<<"#children = "<<node.children.size()<<" and errors= "<<node.error<<" size is "<<(node.data.size())<<" and entropy= "<<node.entropy<<endl;
    else
        cout<<node.feature<<" "<<"#children = "<<node.children.size()<<" and errors= "<<node.error<<" size is "<<(node.data.size())<<" and entropy= "<<node.entropy<<endl;
    
    //for(int j = 0; j<14;j++)
    //cout<<node.feature_done[j]<<" ";
    //cout<<endl<<endl;
    static int passed = 0;
    if(node.children.size()!=0)
    {
        if(passed++ == levels)
            exit(0);
        for(it=node.children.begin();it!=node.children.end();it++)
            print_tree(*(it->second),levels);
        //break;
    }
    
}
map<long long, vector<tr_example> > create_map(vector<tr_example> data, long long i)
{
    vector<tr_example>::iterator data_iter;
    map<long long,vector<tr_example> > mymap;
    map<long long,vector<tr_example> >::iterator my_iter;
    vector<tr_example> data_temp;
    for(data_iter = data.begin();data_iter!=data.end();++data_iter)  //Iterate through all the entries to classify based on value for feature[i]
    {
        my_iter = mymap.find(data_iter->feature[i]);
        if(my_iter != mymap.end())
            my_iter->second.push_back(*data_iter);
        else     //if the value occurs for the first time for this feature
        {
            data_temp.clear();     //data_temp is used to store the row that is the first to have the specific value for feature[i]
            data_temp.push_back(*data_iter);
            mymap.insert(make_pair(data_iter->feature[i],data_temp));
        }
    }
    return mymap;
}


map<long long, vector<tr_example> > create_map_cont(vector<tr_example> data, long long i,double a)
{
    vector<tr_example>::iterator data_iter;
    map<long long,vector<tr_example> > mymap;
    map<long long,vector<tr_example> >::iterator my_iter;
    vector<tr_example> data_temp;
    vector<long long> t;
    for(data_iter = data.begin();data_iter!=data.end();++data_iter)  //Iterate through all the entries to classify based on value for feature[i]
    {
        if((data_iter->feature[i])<a)
            my_iter = mymap.find(0);
        else
            my_iter = mymap.find(1);
        
        if(my_iter != mymap.end())
            my_iter->second.push_back(*data_iter);
        else     //if the value occurs for the first time for this feature
        {
            data_temp.clear();     //data_temp is used to store the row that is the first to have the specific value for feature[i]
            data_temp.push_back(*data_iter);
            if((data_iter->feature[i])<a)
                mymap.insert(make_pair(0,data_temp));
            else
                mymap.insert(make_pair(1,data_temp));
        }
    }
    return mymap;
}

double find_p_entropy(double p0)               // New fucntion check in context of other changed code.
{   double lp=p0,mp=(1-p0),entropy=0;
    if((mp == 0)&&(lp == 0))
        entropy = 0;
    else if (mp == 0)
        entropy = -1.0 * lp * log2(lp);
    else if (lp == 0)
        entropy = (-1.0)*(mp)*log2(mp);
    else
        entropy = (-1.0)*(mp)*log2(mp) - 1.0*(lp)*log2(lp);
    
    return entropy;
}
double find_entropy(vector<tr_example> data)
{
    long long more = 0, less = 0;
    double entropy=0.0;
    vector<tr_example>::iterator data_iter;
    for(data_iter = data.begin(); data_iter!=data.end(); ++data_iter)
    {
        if(data_iter->feature[14] == 1)
            less++;
        else
            more++;
    }
    float mp = (more*1.0)/(data.size() * 1.0);
    float lp = (less*1.0)/(data.size() * 1.0);
    if((more == 0)&&(less == 0))
        entropy = 0;
    else if (more == 0)
        entropy = -1.0 * lp * log2(lp);
    else if (less == 0)
        entropy = (-1.0)*(mp)*log2(mp);
    else
        entropy = (-1.0)*(mp)*log2(mp) - 1.0*(lp)*log2(lp);
    
    return entropy;
}

double find_p(vector<tr_example> data)
{
    long long less = 0, more = 0;
    vector<tr_example>::iterator data_iter;
    for(data_iter = data.begin(); data_iter!=data.end(); ++data_iter)
    {
        if(data_iter->feature[14] == 1)
            less++;
        else
            more++;
    }
    float less_prob = (less*1.0)/(data.size() * 1.0);
    
    return less_prob;
}

long long find_err(vector<tr_example> data)
{
    long long less = 0, more = 0;
    vector<tr_example>::iterator data_iter;
    for(data_iter = data.begin(); data_iter!=data.end(); ++data_iter)
    {
        if(data_iter->feature[14] == 1)
            less++;
        else
            more++;
    }
    if(less>more)
        return more;
    else
        return less;
}
int find_feature(dt_node prev_node,data_unique D,double *thre)
{
    vector<tr_example> data = prev_node.data;
    double prev_entropy = prev_node.entropy;
    *thre=0;
    dt_node answer_node;
    int i, all_done=1,flag_first = 1;
    double curr_entropy = 0.0, max_gain =0.0, gain=0.0,max_index=-1,sinfo=0.0;
    map<long long,vector<tr_example> > mymap,max_map;
    map<long long,vector<tr_example> >::iterator my_iter;
    gain=0;
    map<double,double> a;
    map<double,double>::iterator k,j;
    vector<double> disc;
    vector<double>::iterator itera;
    vector<pair<long long,long long> > count;
    vector<pair<long long,long long> >::iterator ite;
    vector<long long>::iterator itl;
    double p0=0,p1=0,s0=0,s1=0;
    vector<tr_example>::iterator data_iter;
    double max_gain_c=0,max_iter_c=0,flag=1,max_feature_c=0,split=0,max_split=0,dispt=0;

    for(i=0;i<14;i++)  //iterate through all the features to find max gain feature
    {
        //cout<<"Im here"<<endl;
        if(D.cont[i]==0)
        {
            sinfo=0.0;
            curr_entropy = 0.0;
            gain = 0.0;
            if(prev_node.feature_done[i] != 1)
            {
                all_done = 0;
                mymap.clear();   //different for each feature
                    mymap = create_map(data,i);
                //Checking for feature that gives maximum entropy
                curr_entropy = 0;
                for (my_iter = mymap.begin(); my_iter != (mymap.end()); ++my_iter)
                {
                curr_entropy += (float)find_entropy(my_iter->second)*((double)my_iter->second.size()/(double)prev_node.data.size())*1.0;
                if(((double)my_iter->second.size()/(double)prev_node.data.size())!=0)
                    sinfo-=((double)my_iter->second.size()/(double)prev_node.data.size())*log2(((double)my_iter->second.size()/(double)prev_node.data.size()));
                }
                gain = prev_entropy - curr_entropy;
                if(sinfo!=0)
                    gain=gain/sinfo;
                else
                    gain=0;
                //cout<<i<<" "<<gain<<" "<<curr_entropy<<endl;
                if(flag_first == 1)
                {
                    max_map = mymap;
                    max_index = i;
                    max_gain = gain;
                    flag_first = 0;
                }
                else if(gain > max_gain)
                {
                    max_map = mymap;
                    max_index = i;
                    max_gain = gain;
                }
        
            }
        
            cout<<"Gain for feaure "<<i<<" is "<<gain<<endl;
           //cout<<"Max gain is "<<max_gain<<endl;
        }
        
        else
        {   s1=0;s0=0;p0=0;p1=0;gain=0;max_split=0;
            max_feature_c=0;
            max_iter_c = 0;
            max_gain_c = 0;
            count.clear();
            a.clear();
            disc.clear();
            
            for (data_iter=data.begin(); data_iter!=data.end(); data_iter++) {      // Making pair of data of the feature we need to check and the corresponding y values
                a.insert(make_pair(data_iter->feature[i],data_iter->feature[14]));
                count.push_back(make_pair(data_iter->feature[i],data_iter->feature[14]));
            }
            
            k=a.begin();
            k++;
            
            for(j=a.begin();k!=a.end();k++,j++)        // Finding all possible points where the y values change and puching into a vector the thresholds correspondingly
                if(k->second-j->second!=0)
                    disc.push_back((k->first+j->first)/2);
            
            
            
            for(itera=disc.begin();itera!=disc.end();itera++)  // Iterating through all the possible values of dicretization thresholds for the given variable
            { s1=0;s0=0;p0=0;p1=0,flag=1;
                for(ite=count.begin();ite!=count.end();ite++) // s0=No. of points less than threshold, s1>threshold; p0-No of pts<threshold & y=1; p1=No pts>threshold & y=1
                {
                    
                    if(ite->first<(*itera))
                    {s0++;
                        if (ite->second==1)
                            p0++;
                    }
                    else
                    {
                        s1++;
                        if (ite->second==1)
                            p1++;
                    }
                }
                
                //cout<<"S1 "<<s1<<" S0 "<<s0<<endl;
                gain=(s0*find_p_entropy(p0/s0)+s1*find_p_entropy(p1/s1))/(s0+s1); //Entropy formula check if right.
                //cout<<*itera<<" "<<gain<<endl;
                split=find_p_entropy((s0)/(s0+s1));                               // Split formula; check if right
                //cout<<"S1 "<<s1<<" S0 "<<s0<<" Split "<<split<<endl;
                //cout<<*itera<<" "<<gain<<endl;
                if(split!=0)
                    gain=gain/split;
                
                //cout<<*itera<<" "<<gain<<endl<<endl<<endl;
                if(flag == 1)                                                    // finding the best split for the given variable
                {
                    max_split=split;
                    max_feature_c=i;
                    max_iter_c = *itera;
                    max_gain_c = gain*split;
                    
                    flag = 0;
                }
                else if(gain < max_gain_c)
                {
                    max_split=split;
                    max_feature_c=i;
                    max_iter_c = *itera;
                    max_gain_c = gain*split;
                }
                
            }
            
            if(flag_first == 1)                                                 // finding the best variable overall.
            {   dispt=max_iter_c;
                max_index = i;
                max_gain = (prev_entropy-max_gain_c)/max_split;
                flag_first = 0;
            }
            else if((((prev_entropy-max_gain_c)/max_split) > max_gain)&& max_split!=0)
            {
                dispt=max_iter_c;
                max_index = i;
                max_gain = (prev_entropy-max_gain_c)/max_split;
            }
           //cout<<"Gain for feaure "<<i<<" is "<<(prev_entropy-(max_gain_c))/max_split<<" split is "<<max_split<<endl;
           // cout<<"Max gain is "<<max_gain<<endl;
        }
        
    }
    
    if((max_gain<=0)||(all_done)||(curr_entropy<=0.7))   //Check and change this value of limit on the curr_entropy<0.7 to curr_entropy==0 once you manually prune
    {
        //cout<<"Correct"<<endl;
        return -1;
    }
    cout<<max_index<<endl;
    *thre=dispt;
    return max_index;
    
    
}

double risk_calc(dt_node node,int a=1)
{
    static long long R=0,count=0;
    if(a==1)
    {
        R=0;
        count=0;
    }
    
    if(node.children.size()==0)
    {
        R=R+(long long)node.error;
        //cout<<"Adding "<<*R<<endl;
        count++;
    }
    else
    {
        //cout<<"Visited "<<dt_root.feature<<endl;
        map<long long, dt_node*>::iterator children_iter;
        for(children_iter = node.children.begin();children_iter!=node.children.end();++children_iter)
        {
            //cout<<(children_iter->second)->parent->feature<<endl;
            risk_calc(*(children_iter->second),0);
        }
    }
    
    //cout<<"The risk calculator was called and returned value "<<((double)(R)/(double)(N))+(double)(ALPHA*(count))<<endl;
    return ((double)(R)/(double)(N))+(ALPHA*(count));
}


dt_node divide(dt_node node,data_unique D)
{   double thre=0;
    //dt_node node = *node_pointer;
    cout<<"Calling Divide"<<endl;
    int i=find_feature(node,D,&thre);
    node.feature=i;
    
    if(node.feature == -1)
    {
        return node;
    }
    cout<<"The fearure found is "<<i<<" threshold is "<<thre<<endl;    //cout<<"Dividing based on "<<i<<endl;
    map<long long,vector<tr_example> > mymap,max_map;
    map<long long,vector<tr_example> >::iterator my_iter;
    vector<tr_example>::iterator data_iter;
    vector<tr_example> data_temp;
    vector <tr_example> data=node.data;
    mymap.clear();   //different for each feature
    if(D.cont[i]==0)
        mymap = create_map(data,i);
    else
        mymap = create_map_cont(data,i,thre);
    
    
    for(my_iter=mymap.begin();my_iter!=mymap.end();my_iter++)
    {
        //print_x(my_iter->second,10);
        dt_node* a = new dt_node;
        a->data= my_iter->second;
        a->entropy=find_entropy(my_iter->second);
        a->parent=&node;
        //cout<<a->parent->feature<<endl;
        a->p=find_p(a->data);
        a->feature = -1;
        a->error=find_err(a->data);
        for(int j =0;j<14;j++)
        {
            a->feature_done[j] = node.feature_done[j];
        }
        a->feature_done[node.feature] = 1;
        node.children.insert(make_pair(my_iter->first,a));
    }
    
     if(node.entropy == 0)
     {
     cout<<"-1 due to entropy ";
     node.feature = -1;
     node.children.clear();
     }
    cout<<"Divide Done"<<endl;
    return node;
}

dt_node tree_construct(dt_node node,vector<double>* risks,data_unique D)
{
    node = divide(node,D);
    
    if(node.feature==-1)
        return node;
   
    //risk_calc(dt_root,&R,&count);
    (*risks).push_back(risk_calc(node));
    if(risk_calc(node) < THRESHOLD)
        return node;
    map<long long, dt_node*>::iterator children_iter;
    for(children_iter = node.children.begin();children_iter!=node.children.end();++children_iter)
    {
        //cout<<(children_iter->second)->parent->feature<<endl;
        *(children_iter->second) = tree_construct(*(children_iter->second),risks,D);
    }
    return node;
}


double traverse(tr_example a,dt_node node)
{
    double p;
   // cout<<"This node has feature "<<node.feature<<endl;
   // node.
    //if(node.children.size()==0)
    //{
        
    //}
    if(node.children.size()==0)
    {
        //cout<<"ENDED AT "<<dt_root.feature<<endl;
        return node.p;
    }
    else
    {
        
        map<long long, dt_node*>::iterator children_iter = node.children.find(a.feature[node.feature]);
        /*
        for(children_iter = node.children.begin();children_iter!=node.children.end();++children_iter)
        {
            if(children_iter->first==a.feature[node.feature])
            {
                //cout<<"Found feature"<<endl;
                p = traverse(a,*(children_iter->second));
                return p;
            }
        }
         */
        if(children_iter != node.children.end())
        {
            p = traverse(a,*(children_iter->second));
            return p;
        }
        else
        {
            return node.p;
        }
    }
    //cout<<"Search for "<<node.feature<<" th feature = "<<a.feature[node.feature]<<" map size = "<<node.children.size();
    //cout<<" Returning zero"<<endl;
    return 0;
}


dt_node create_dtree(data_unique tr_dat)
{
    double thre;
    vector<double> risks;
    vector<double>::iterator risk_iter;
    risks.clear();
    //Creating and initializing root node
    
    dt_root.data = tr_dat.D;
    dt_root.entropy=find_entropy(dt_root.data);
    dt_root.value = -1;
    dt_root.p=find_p(dt_root.data);
    dt_root.error=find_err(dt_root.data);
    //cout<<"Root error calculated "<<dt_root.error<<endl;
    for(int i =0;i<14;i++)
        dt_root.feature_done[i] = tr_dat.cont[i];
    
   /* dt_root.feature_done[0] = 1;
    dt_root.feature_done[2] = 1;
    dt_root.feature_done[4] = 1;
    dt_root.feature_done[12] = 1;
    dt_root.feature_done[10] = 1;
    dt_root.feature_done[11] = 1;*/
    dt_root.parent = NULL;
    
    risks.push_back(risk_calc(dt_root));
    cout<<"Tree growing.."<<endl;
    //cout<<"The fearure found is "<<find_feature(dt_root, tr_dat,&thre)<<" threshold is "<<thre<<endl;
    dt_root=divide(dt_root,tr_dat);
    //map<long long, dt_node*>::iterator children_iter;
    //for(children_iter = dt_root.children.begin();children_iter!=dt_root.children.end();++children_iter)
    //{
        //cout<<(children_iter->second)->parent->feature<<endl;
    //    *(children_iter->second) = divide(*(children_iter->second),tr_dat);
    //}
    
    //print_tree(dt_root, 100);
    dt_root=tree_construct (dt_root,&risks,tr_dat);
    //int j=0;
    //cout<<"Size of risk vector "<<risks.size()<<endl;
    //for(risk_iter=risks.begin(),j=0;risk_iter!=risks.end()&&j<2500;risk_iter++,j++)
    //    cout<<*risk_iter<<" ";
    return dt_root;
}


int main()
{
    //Initializations
    
   
    map<long long, dt_node*>::iterator children_iter;
    vector<tr_example> tr_data;
    vector<tr_example>::iterator data_iter;
    char input[100];
    int predicted = 1;
    
    //Getting the input data from get_data
    
    cin>>input;
    data_unique training_data=get_data(input);
    N=training_data.n;
    
    //print_x(training_data.D, 10);
    
    dt_root=create_dtree(training_data);
    
    print_tree(dt_root,1000);
    
    
    //cout<<"Root Risk"<<risk_calc(dt_root)<<" "<<endl;
    
    //dt_root = divide(dt_root,training_data);
    //print_tree(dt_root,10000);
    //cout<<"Root Risk"<<risk_calc(dt_root)<<" "<<endl;
    
    cout<<"Enter Test input file location"<<endl;
    char inputt[100];
    cin>>inputt;
    cout<<endl;
    //cout<<"Enter the path for the testing data"<<endl;
    data_unique testing_data=get_data(inputt,training_data);
    //print_x(testing_data.D,10);
    
    vector<tr_example> test_data = testing_data.D;
    long long corr_pred = 0;
    long long wr_pred = 0;
    int lim=0;
    
    for(data_iter = test_data.begin(),lim=0;data_iter != test_data.end()&& lim<1000; ++data_iter,lim++)
    {
     //data_iter = test_data.begin();
     predicted = ((float)traverse(*data_iter,dt_root)<0.5)?2:1;
     //cout<<traverse(*(dt_root.data.begin()),dt_root)<<"  "<<(dt_root.data.begin())->feature[14]<<endl;
     //cout<<"Predicted "<<predicted<<" Actual "<<data_iter->feature[14]<<endl;
     if (predicted == data_iter ->feature[14])
     corr_pred++;
     else
     wr_pred++;
     }
     
     cout<<"correct "<<corr_pred<<" Wrong "<<wr_pred<<endl;
     cout<<"Accuracy = "<<(float)corr_pred/(float)(corr_pred + wr_pred)<<endl;
    
    return 0;
    
}

