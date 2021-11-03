#include <iostream>
#include <Dense>
#include <vector>

int main()
{

    Eigen::ArrayXXd a(3, 3), b(2,2);
    a << 1,2,3,4,5,6,7,8,9;
    b<< 1,2,3,4;
    std::vector<int> ind({0,2});
    std::cout<<a<<std::endl;
    a(ind, ind) +=b;
    std::cout<<a<<std::endl;
    

    return 0;
}