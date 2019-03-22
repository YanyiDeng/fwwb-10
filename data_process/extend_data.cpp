#include <bits/stdc++.h>
#define num 700 
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
static const int maxn = 100010;
static const int INF = 0x3f3f3f3f;
static const int mod = (int)1e9 + 7;
static const double eps = 1e-6;
static const double pi = acos(-1);

void redirect(){
    #ifdef LOCAL
        freopen("../raw_data/train_8.tsv","r",stdin);
        freopen("../raw_data/newtrain.tsv","a+",stdout);
    #endif
}
map<string,vector<string> >m;
vector<pair<int,string> >v;
int main(){
    redirect();
    string str;
    getline(cin,str);
    while(getline(cin,str)){
        int pos = str.find('\t')+1;
        m[string(&str[pos])].push_back(str);
    }
    /*for(auto it = m.begin();it != m.end();it++)
    sz = max(sz,(int)(it->second).size());*/
    printf("ITEM_NAME	TYPE\n");
    for(auto it = m.begin();it != m.end();it++){
        int cnt = (it->second).size();
        int sz = max(num,cnt);
        for(int i = 1;i <= sz/cnt;i++)
        for(auto x : it->second)
        printf("%s\n",&x[0]);
        for(int i = 0;i < sz%cnt;i++)
        printf("%s\n",&(it->second)[i][0]);
    }
    /*for(auto it = m.begin();it != m.end();it++)
    v.push_back(make_pair((it->second).size(),it->first));
    sort(v.begin(),v.end());
    for(auto& it : v){
        cout << " " << it.first << endl;
    }*/ 
    return 0;
}

