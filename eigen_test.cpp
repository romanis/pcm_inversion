#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXi m(1, 5);
    m << 1, 2, 3, 4, 5;
    m = (m.array() > 3);
    std::cout << m << std::endl;

    return 0;
}