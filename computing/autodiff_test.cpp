
// C++ includes
#include <iostream>
using namespace std;
#include "eigen3/Eigen/Core"
#include "autodiff_test_2.h"
#include <eigen3/unsupported/Eigen/AutoDiff>


/**
 * A sample function which has matrix valued arguments and
 * a scalar result. Again the function is templated so it
 * can be used with either normal floating point number
 * and AutoDiffScalar.
 */
template <typename T>
T my_matrixfun(Eigen::Matrix<T, 2, 1> const &a)
{
    Eigen::Matrix<T, 2, 1> gain;
    for (auto row = 0; row < gain.rows(); ++row){gain(row) = 2;};
    return ( a.transpose() * gain.asDiagonal() * a)(0,0);
}

/**
 * Extract the gradient from my_matrixfun()
// */
//void test_matrix(){
//    std::cout << "== test_matrix() ==" << std::endl;
//    // use with normal floats
//
//    Eigen::Matrix<double, -1, 1> a(4); a << 1, 2, 3, 4;
//    Eigen::Matrix<double, -1, 1> b(4); b << 3, 4, 5, 6;
//
//    double c = my_matrixfun(a,b);
//    std::cout << "Result: " << c << std::endl;
//
//    // use with AutoDiffScalar
//    using AScalar_a = Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1>>;
//    using AScalar_b = Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1>>;
//    using AVector_a = Eigen::Matrix<AScalar_a, -1, 1>;
//    using AVector_b = Eigen::Matrix<AScalar_b, -1, 1>;
//    AVector_a Aa(4); AVector_b Ab(4);
//    // copy value from non-active example
//    for(int i=0;i<a.size();i++)Aa(i).value() = a(i);
//    for(int i=0;i<b.size();i++)Ab(i).value() = b(i);
//    // initialize derivative vectors
//    const int derivative_num = a.size() + b.size();
//    int derivative_idx = 0;
//    for(int i=0;i<Aa.size();i++){
//        Aa(i).derivatives() =
//                Eigen::VectorXd::Unit(derivative_num, derivative_idx);
//        derivative_idx++;
//    }
//    for(int i=0;i<Ab.size();i++){
//        Ab(i).derivatives() =
//                Eigen::VectorXd::Unit(derivative_num, derivative_idx);
//        derivative_idx++;
//    }
//
//    AScalar_a Ac = my_matrixfun(Aa,Ab);
//    std::cout << "Result: " << Ac.value() << std::endl;
//    std::cout << "Gradient: " <<
//              Ac.derivatives().transpose() << std::endl;
//}

/**
 * Tag an twice active scalar with a given derivative direction.
 */
template <typename T>
void init_twice_active_var(T &ad,int d_num, int idx){
    // initialize derivative direction in value field of outer active variable
    ad.value().derivatives() = T::DerType::Scalar::DerType::Unit(d_num,idx);
    // initialize derivatives direction of the variable
    ad.derivatives() = T::DerType::Unit(d_num,idx);
    // initialize Hessian matrix of variable to zero
    for(int index=0; index < d_num; index++){
        ad.derivatives()(index).derivatives()  = T::DerType::Scalar::DerType::Zero(d_num);
    }
}

/**
 * Generating the gradient and the hessian from my_matrixfun
 */
void test_matrix_twice(){
    std::cout << "== test_matrix_twice() ==" << std::endl;
    // use with normal floats

    Eigen::Matrix<double, 2, 1> a; a << 1, 2;

    double c = my_matrixfun(a);

    // use with AutoDiffScalar
    typedef Eigen::Matrix<double, 2,1> inner_derivative_type;
    typedef Eigen::AutoDiffScalar<inner_derivative_type> inner_active_scalar;
    typedef Eigen::Matrix<inner_active_scalar,2,1> outer_derivative_type;
    typedef Eigen::AutoDiffScalar<outer_derivative_type> outer_active_scalar;
    typedef Eigen::Matrix<outer_active_scalar,2,1> AVector;
    AVector Aa(a.size());
    // copy value from non-active example
    for(int i=0;i<a.size();i++)Aa(i).value().value() = a(i);
//    for(int i=0;i<b.size();i++)Ab(i).value().value() = b(i);
    // initialize derivative vectors
    const int derivative_num = a.size();
    int derivative_idx = 0;
    for(int i=0;i<Aa.size();i++){
        init_twice_active_var(Aa(i),derivative_num,derivative_idx);
        derivative_idx++;
    }
//    for(int i=0;i<Ab.size();i++){
////        init_twice_active_var(Ab(i),derivative_num,derivative_idx);
////        derivative_idx++;
////    }

    outer_active_scalar Ac = my_matrixfun(Aa);
    std::cout << "Result: " << Ac.value().value() << std::endl;
    std::cout << "Gradient: " <<
              Ac.value().derivatives().transpose() << std::endl;

    Eigen::Matrix2d hessian(Ac.derivatives().size(),Ac.derivatives().size());
    for(int idx=0;idx<Ac.derivatives().size();idx++){
        hessian.middleRows(idx,1) =
                Ac.derivatives()(idx).derivatives().transpose();
    }
    std::cout << "Hessian" << "\n" << hessian << std::endl;
}


int main()
{
//    CostFunction cost(&running, &terminal);
//
//    VectorXdual x(4);
//    x << 1, 2, 3, 4;
//
//    VectorXdual u(2);
//    u << 1, 2;
//
//    cost.update_state(x, u);
//    auto result = cost.L_xx();
//    std::cout << result << std::endl;


/**
 * This describes the basic use of Eigen::AutoDiffScalar
 */


	// basic stuff
//    basic_use_autodiff_scalar();
//    test_scalar();
//    test_matrix();
//
//    // advanced stuff
    test_matrix_twice();
//    return 0;

}