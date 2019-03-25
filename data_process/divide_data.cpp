#include <bits/stdc++.h>
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
        freopen("../raw_data/train.tsv","r",stdin);
    #endif
}
map<string,vector<string> >m;
set<int>s;
int main(){
    redirect();
    string str;
    getline(cin,str);
    freopen("../raw_data/train_8.tsv","a+",stdout);
    cout << str << endl;
    freopen("../raw_data/train_2.tsv","a+",stdout);
    cout << str << endl;
    while(getline(cin,str)){
        int pos = str.find('\t')+1;
        m[string(&str[pos])].push_back(str);
    }
    srand(time(0));
    for(auto it = m.begin();it != m.end();it++){
        s.clear();
        int sz = (it->second).size();
        freopen("../raw_data/train_8.tsv","a+",stdout);
        for(int i = 0;i < 0.8*sz;i++){
            int id = rand()%sz;
            if(s.count(id)){
                i--;
                continue;
            }
            else{
                s.insert(id);
                cout << (it->second)[id] << endl;
            }
        }
        freopen("../raw_data/train_2.tsv","a+",stdout);
        for(int i = 0;i < sz;i++)
        if(!s.count(i))cout << (it->second)[i] << endl;
    }
    return 0;
}

