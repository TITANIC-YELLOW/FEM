#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <iomanip>

const double const1 = 1.0 / std::pow(3.0, 0.5);

const double a27 = -std::sqrt(0.6);
const double b27 = 0;
const double c27 = std::sqrt(0.6);

const double d27 = std::pow(5.0 / 9.0, 3);
const double e27 = (8.0 / 9.0) * std::pow(5.0 / 9.0, 2);
const double f27 = std::pow(8.0 / 9.0, 2) * (5.0 / 9.0);
const double g27 = std::pow(8.0 / 9.0, 3);


double HexaShapeFunction(int order, double x, double y, double z) {
    Eigen::Matrix<double, 8, 3> CoefficientMatrix;
    CoefficientMatrix<<-1, -1, -1,
                        1, -1, -1,
                        1,  1, -1,
                       -1,  1, -1,
                       -1, -1,  1,
                        1, -1,  1,
                        1,  1,  1,
                       -1,  1,  1;
    double result;
    result = (1 + CoefficientMatrix(order, 0) * x) * (1 + CoefficientMatrix(order, 1) * y) * 
             (1 + CoefficientMatrix(order, 2) * z) * 0.125;
    return result;
}

class hexahedron {
private:
    Eigen::Matrix<double, 8, 3> HexaIntPoints;
    Eigen::Matrix<double, 8, 3> CoordinateArray;
    
public:
    hexahedron(Eigen::Matrix<double, 8, 3> coordinate);
    double volume();
    double DetOfHexa(double x, double y, double z);
    //double IntInHexahedron(double(&func)(double, double, double));
    Eigen::Matrix<double, 8, 8> mass_matrix(double p, double c);
};

hexahedron::hexahedron(Eigen::Matrix<double, 8, 3> coordinate) {
    CoordinateArray = coordinate;
    HexaIntPoints << -const1, -const1, -const1,
                     -const1, -const1,  const1,
                     -const1,  const1, -const1,
                     -const1,  const1,  const1,
                      const1, -const1, -const1,
                      const1, -const1,  const1,
                      const1,  const1, -const1,
                      const1,  const1,  const1;
}


double hexahedron::DetOfHexa(double x,double y,double z) {
    double ans;
    Eigen::Matrix<double, 3, 8> dN;
    Eigen::Matrix<double, 3, 3> JacobiMatrix;

    dN(0, 0) = (-1) * (1 - y) * (1 - z);
    dN(0, 1) = (1 - y) * (1 - z);
    dN(0, 2) = (1 + y) * (1 - z);
    dN(0, 3) = (-1) * (1 + y) * (1 - z);
    dN(0, 4) = (-1) * (1 - y) * (1 + z);
    dN(0, 5) = (1 - y) * (1 + z);
    dN(0, 6) = (1 + y) * (1 + z);
    dN(0, 7) = (-1) * (1 + y) * (1 + z);

    dN(1, 0) = (-1) * (1 - x) * (1 - z);
    dN(1, 1) = (-1) * (1 + x) * (1 - z);
    dN(1, 2) = (1 + x) * (1 - z);
    dN(1, 3) = (1 - x) * (1 - z);
    dN(1, 4) = (-1) * (1 - x) * (1 + z);
    dN(1, 5) = (-1) * (1 + x) * (1 + z);
    dN(1, 6) = (1 + x) * (1 + z);
    dN(1, 7) = (1 - x) * (1 + z);

    dN(2, 0) = (-1) * (1 - y) * (1 - x);
    dN(2, 1) = (-1) * (1 - y) * (1 + x);
    dN(2, 2) = (-1) * (1 + y) * (1 + x);
    dN(2, 3) = (-1) * (1 + y) * (1 - x);
    dN(2, 4) = (1 - y) * (1 - x);
    dN(2, 5) = (1 - y) * (1 + x);
    dN(2, 6) = (1 + y) * (1 + x);
    dN(2, 7) = (1 + y) * (1 - x);

    JacobiMatrix = dN * CoordinateArray * 0.125;
    ans = std::abs(JacobiMatrix.determinant());

    return ans;
}


double hexahedron::volume() {

    double ans = 0.0;
    
    double x, y, z;

    for (int i = 0; i <= 7; ++i) {
        x = HexaIntPoints(i, 0);
        y = HexaIntPoints(i, 1);
        z = HexaIntPoints(i, 2);

        ans += DetOfHexa(x, y, z);
    }
    return ans;
}


Eigen::Matrix<double, 8, 8> hexahedron::mass_matrix(double p, double c) {
    Eigen::Matrix<double, 8, 8> m = Eigen::Matrix<double, 8, 8>::Zero();
    double x, y, z, det;

    for (int i = 0; i <= 7; ++i) {
        x = HexaIntPoints(i, 0);
        y = HexaIntPoints(i, 1);
        z = HexaIntPoints(i, 2);
        det = DetOfHexa(x, y, z);

        for (int row = 0; row <= 7; ++row) {
            for (int col = 0; col <= 7; ++col) {
                double N1 = HexaShapeFunction(row,x,y,z);
                double N2 = HexaShapeFunction(col,x,y,z);
                m(row, col) += N1 * N2 * det;
            }
        }
    }

    m = m * p * c;
    return m;
}

//double hexahedron::IntInHexahedron(double (&func)(double, double, double)) {
//
//    double ans = 0.0;
//    Eigen::Matrix<double, 8, 3> HexaIntPoints;
//    HexaIntPoints << -const1, -const1, -const1,
//                     -const1, -const1,  const1,
//                     -const1,  const1, -const1,
//                     -const1,  const1,  const1,
//                      const1, -const1, -const1,
//                      const1, -const1,  const1,
//                      const1,  const1, -const1,
//                      const1,  const1,  const1;
//    double x, y, z;
//
//    for (int i = 0; i <= 7; ++i) {
//        x = HexaIntPoints(i, 0);
//        y = HexaIntPoints(i, 1);
//        z = HexaIntPoints(i, 2);
//
//        ans += func(x, y, z) ;
//        }
//    return ans;
//}




//double IntInPyramid(double (*func)(double, double, double)) {
//    double ans = 0.0;
//
//    double x, y, z, const1, const2;
//
//    for (int i = 0; i <= 3; ++i) {
//        for (int j = 0; j <= 26; ++j) {
//
//            x, y, z, const1, const2 = pyramidlist[i][j];
//            ans += func(x, y, z) * const1 * const2;
//        }
//    }
//    return ans;
//}
int main() {
    Eigen::Matrix<double, 8, 3> arr;
    arr << 0, 0, 0,
           1, 0, 0,
           1, 1, 0,
           0, 1, 0,
           0, 0, 1,
           1, 0, 1,
           1, 1, 1,
           0, 1, 1;

    hexahedron object(arr);
    std::cout<<std::setprecision(15)<<object.volume()<<std::endl;
    std::cout << object.mass_matrix(1, 1) << std::endl;
}

